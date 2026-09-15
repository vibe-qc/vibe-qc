"""Nuclear-density smearing infrastructure for the v0.10.x GAPW route.

> **Experimental, infrastructure-only.** Built for M3b's GAPW
> augmentation work (gapw chat, audit-driven). Not yet wired into
> ``evaluate_gpw_energy`` or ``run_periodic_rhf_gpw`` -- the M3a SCF
> still uses the Ewald-lattice V_ne path. The decision between the
> two V_ne conventions (D5 in ``docs/design_periodic_gapw.md``) is
> pending the maintainer's call.

CP2K-style nuclear charge smearing: each point nucleus ``Z_a`` at
position ``R_a`` is replaced by a normalised Gaussian

    ``r_core_a(r) = Z_a . (a/√pi)^3 . exp(-a^2 . |r - R_a|^2)``

so that the full ``S_a Z_a . 1/|r - R_a|`` nuclear potential splits as

    ``Z_a / |r - R_a|  =  Z_a . erf(a |r - R_a|) / |r - R_a|
                      +  Z_a . erfc(a |r - R_a|) / |r - R_a|``

The first piece is exactly the Hartree potential of ``r_core_a``;
it is smooth, band-limited, and can be solved on the FFT grid via
the same Poisson kernel as the electronic Hartree. The second piece
is short-range (decays inside ``~3/a`` bohr) and is handled either
analytically on a per-atom radial grid (M3b GAPW augmentation) or
via libint's `erfc_coulomb` one-electron operator (M3b GPW
pseudo-potential mode, if we go that way).

This module provides the **smooth-grid side** of that split: the
nuclear density on the FFT grid, the spurious self-Hartree
correction, and the Ewald-style pairwise overlap formula for the
nuclear-nuclear interaction. M3b consumes these to build a V_ne
matrix and an E_nn correction that share the same gauge as the
electronic FFT-Poisson Hartree.

Convention map vs. ``periodic_gapw_grid.collocate_point_charges_on_grid``:

* the existing M1d primitive parameterises Gaussians by *real-
  space width* ``s`` (the standard normal-distribution s);
* CP2K and the textbook Ewald derivations parameterise by
  *reciprocal-space exponent* ``a`` such that
  ``r = (a/√pi)^3 exp(-a^2 r^2)``;
* the two relate by ``s = 1 / (a √2)``, equivalently ``a = 1 /
  (s √2)``.

This module exposes the a convention publicly and converts to s
internally when delegating to the M1d primitive -- so callers stay
calibration-consistent with CP2K's ``zet0_h``, the Ewald-a in
``EwaldOptions``, and the FFT-Poisson convergence note in the CP2K
audit ("``a . REL_CUTOFF ≪ CUTOFF`` to avoid aliasing").

Audit cross-references:

* **D1** (``docs/design_periodic_gapw.md``): the multipole
  compensator r_0 sits on top of this smeared nuclear density;
  not built here, but the architecture in this module is the
  natural home for it.
* **D5**: V_ne convention decision. This module supplies the
  pieces needed by the "CP2K erfc/erf split" path.
* CP2K source comments (`qs_rho0_methods.F`, `init_rho0`,
  `calculate_ecore_self`) for the conventions paraphrased here --
  no GPL source quoted, just the algorithm.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import _vibeqc_core as _core
from .periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
    collocate_point_charges_on_grid,
)

__all__ = [
    "SmearedNuclearCharges",
    "default_smearing_alpha_from_grid",
    "smeared_nuclear_density_on_grid",
    "smeared_self_energy",
    "smeared_pairwise_overlap_energy",
    "smeared_v_ne_long_matrix",
    "smeared_v_ne_erfc_short_matrix",
    "smeared_v_ne_full_gamma",
    "alpha_to_sigma",
    "sigma_to_alpha",
    "GaussianMultipoleCompensator",
    "monopole_compensator_for_charge",
    "compute_atom_local_multipoles",
    "build_electronic_compensator",
]


# ============================================================
# Parameterisation conventions
# ============================================================


def alpha_to_sigma(alpha: float) -> float:
    """Convert CP2K-style ``a`` (reciprocal-space Gaussian exponent
    in 1/bohr) to ``s`` (real-space standard deviation in bohr).

    Relationship: ``r = (a/√pi)^3 exp(-a^2 r^2) = (2pi s^2)^(-3/2) .
    exp(-r^2 / (2 s^2))`` => ``s = 1 / (a √2)``.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    return 1.0 / (alpha * math.sqrt(2.0))


def sigma_to_alpha(sigma: float) -> float:
    """Inverse of :func:`alpha_to_sigma`."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    return 1.0 / (sigma * math.sqrt(2.0))


# ============================================================
# Default a heuristic
# ============================================================


def default_smearing_alpha_from_grid(
    grid: PlaneWaveGrid,
    *,
    eps_tail: float = 1.0e-10,
) -> float:
    """Pick ``a`` so that the smeared nucleus is well-resolved by
    ``grid``.

    The reciprocal-space form of a normalised Gaussian density of
    exponent ``a`` is ``r̂(G) = exp(-|G|^2 / (4 a^2))``. To keep
    aliasing of r_core on the FFT grid below ``eps_tail``, we need

        ``|G|^2_max / (4 a^2)  >=  -ln(eps_tail)``

    where ``|G|_max = √(2 . E_cut)`` for a kinetic-energy cutoff
    ``E_cut`` in Hartree. Solving for the largest safe ``a``:

        ``a <= √( E_cut / (2 . ln(1/eps_tail)) )``

    For the typical M2 demo cutoff ``E_cut = 300 Ha`` and
    ``eps_tail = 1e-10`` (≈ -23.0 ln), this gives ``a ≲ 2.55``
    bohr⁻¹. The CP2K rule of thumb ``a . REL_CUTOFF ≪ CUTOFF`` is
    the same inequality dressed differently.

    Raises
    ------
    ValueError
        If ``grid.cutoff_ha`` is ``None`` (grid built without going
        through the cutoff heuristic) -- the heuristic needs a
        cutoff to pick a. Caller must supply a directly in that
        case.
    """
    if grid.cutoff_ha is None:
        raise ValueError(
            "default_smearing_alpha_from_grid: grid was built without "
            "a cutoff_ha. Either rebuild via make_grid(..., cutoff_ha=...) "
            "or pass alpha explicitly to the SmearedNuclearCharges "
            "constructor."
        )
    if eps_tail <= 0.0 or eps_tail >= 1.0:
        raise ValueError(
            f"eps_tail must be in (0, 1); got {eps_tail!r}"
        )
    log_inv_eps = -math.log(eps_tail)
    alpha = math.sqrt(grid.cutoff_ha / (2.0 * log_inv_eps))
    return alpha


# ============================================================
# Per-atom smeared-charge bundle
# ============================================================


@dataclass(frozen=True)
class SmearedNuclearCharges:
    """Bundle of (position, Z, a) per atom, the inputs to the M3b
    smeared-nuclear V_ne / E_nn path.

    Per-atom ``alpha`` is permitted in the dataclass so element-
    specific smearing widths (CP2K's per-`KIND` `zet0_h`) compose
    naturally. The default factory
    :func:`from_periodic_system_uniform_alpha` builds a uniform-a
    bundle from a :class:`PeriodicSystem`.

    Attributes
    ----------
    positions
        ``(n_atoms, 3)`` Cartesian positions in bohr.
    Z
        ``(n_atoms,)`` nuclear charges (positive). These are the
        atomic numbers for bare nuclei or the effective core
        charges for pseudopotentials.
    alpha
        ``(n_atoms,)`` reciprocal-space Gaussian exponents in
        bohr⁻¹. A uniform a is the common case;
        :meth:`__post_init__` enforces ``alpha > 0`` per entry.
    """

    positions: np.ndarray
    Z: np.ndarray
    alpha: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (n_atoms, 3); got shape "
                f"{positions.shape}"
            )
        n_atoms = positions.shape[0]
        Z = np.asarray(self.Z, dtype=float)
        if Z.shape != (n_atoms,):
            raise ValueError(
                f"Z shape {Z.shape} doesn't match positions "
                f"({n_atoms} atoms)"
            )
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.shape != (n_atoms,):
            raise ValueError(
                f"alpha shape {alpha.shape} doesn't match positions "
                f"({n_atoms} atoms)"
            )
        if not np.all(alpha > 0.0):
            raise ValueError("alpha entries must all be positive")
        # Persist canonical numpy views; the dataclass is frozen so
        # use object.__setattr__.
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "Z", Z)
        object.__setattr__(self, "alpha", alpha)

    @classmethod
    def from_periodic_system_uniform_alpha(
        cls,
        system,
        alpha: float,
    ) -> "SmearedNuclearCharges":
        """Build from a :class:`PeriodicSystem` with a single a
        shared by every atom.

        Use this when you want one Ewald-style smearing for the
        whole cell -- the typical M3b first cut. For per-element a
        (CP2K-style `zet0_h`), use the dataclass constructor
        directly.
        """
        positions = np.array(
            [list(a.xyz) for a in system.unit_cell], dtype=float
        )
        Z = np.array(
            [float(int(a.Z)) for a in system.unit_cell], dtype=float
        )
        alpha_arr = np.full(positions.shape[0], float(alpha), dtype=float)
        return cls(positions=positions, Z=Z, alpha=alpha_arr)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    @property
    def total_charge(self) -> float:
        """``S_a Z_a`` -- the integrated nuclear charge."""
        return float(self.Z.sum())

    @property
    def is_uniform_alpha(self) -> bool:
        """True iff every atom shares the same a (within 1e-14)."""
        if self.n_atoms == 0:
            return True
        return bool(np.all(np.abs(self.alpha - self.alpha[0]) < 1e-14))


# ============================================================
# Density on the FFT grid
# ============================================================


def smeared_nuclear_density_on_grid(
    charges: SmearedNuclearCharges,
    grid: PlaneWaveGrid,
    *,
    image_shells: int = 1,
) -> np.ndarray:
    """Lay down ``r_core(r) = S_a Z_a . (a_a/√pi)^3 exp(-a_a^2 . |r -
    R_a|^2)`` on the FFT grid.

    Delegates per-atom to :func:`collocate_point_charges_on_grid`
    after converting each a to the s convention; permits per-atom
    a by looping over atoms one at a time.

    Parameters
    ----------
    charges
        :class:`SmearedNuclearCharges` bundle.
    grid
        :class:`PlaneWaveGrid` defining the cell + sampling.
    image_shells
        Number of neighbouring-image shells to sum (default 1 ->
        27 cells). The Gaussian tail at ``image_shells=1`` for a
        typical ``a ~ 2`` bohr⁻¹ in an 8-bohr cube is ``~exp(-65)``,
        i.e., zero to machine precision.

    Returns
    -------
    rho
        ``(nx, ny, nz)`` real-space density on the grid, in
        ``e / bohr^3``. Integrates to ``S_a Z_a`` within the
        finite-cell quadrature noise.
    """
    rho = np.zeros(grid.shape, dtype=float)
    for i in range(charges.n_atoms):
        sigma = alpha_to_sigma(float(charges.alpha[i]))
        rho += collocate_point_charges_on_grid(
            charges=[float(charges.Z[i])],
            positions_bohr=charges.positions[i:i + 1],
            grid=grid,
            sigma_bohr=sigma,
            image_shells=image_shells,
        )
    return rho


# ============================================================
# Energetic corrections -- self and pairwise overlap
# ============================================================


def smeared_self_energy(charges: SmearedNuclearCharges) -> float:
    """Return the spurious per-atom self-Hartree
    ``S_a Z_a^2 . a_a / √(2pi)`` of the smeared nuclear density.

    The Hartree integral of ``r_core_a`` against itself contains
    ``Z_a^2`` worth of *self*-interaction that has no physical
    counterpart in the bare ``S_a Z_a . 1/|r - R_a|`` nuclear
    potential. Any total-energy bookkeeping that uses the
    FFT-Poisson on r_core must subtract this term.

    For a uniform-a bundle this equals
    ``S_a Z_a^2 . a / √(2pi)``; in the s convention used by
    :func:`vibeqc.periodic_gapw_grid.point_charge_self_energy` it
    is ``S_a Z_a^2 / (2 s_a √pi)`` -- algebraically identical (see
    :func:`alpha_to_sigma`).
    """
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return float((charges.Z ** 2 * charges.alpha * inv_sqrt_2pi).sum())


def smeared_pairwise_overlap_energy(
    charges: SmearedNuclearCharges,
    system,
    *,
    image_shells: int = 1,
) -> float:
    """Pairwise nuclear-nuclear Hartree energy of the *smeared*
    charges, summed over the home cell and ``image_shells`` of
    neighbouring images.

    For two normalised Gaussian densities of exponents ``a_a, a_b``
    at positions ``R_a, R_b``, the Coulomb integral is

        ``Z_a Z_b . erf(b_ab . r_ab) / r_ab``

    where ``b_ab = a_a . a_b / √(a_a^2 + a_b^2)`` (the standard
    Gaussian-product-of-densities convolution). The ``r_ab -> 0``
    limit returns ``Z_a Z_b . 2 b_ab / √pi`` (the smeared analogue
    of the bare ``1/r`` singularity).

    This is the **erf** complement of the true ``1/r`` Coulomb
    nuclear-nuclear interaction; the **erfc** complement is what
    the M3b short-range correction adds back. Returning the sum
    here so an M3b ``E_nn_smeared(charges)`` can decompose into
    "this + Ewald-erfc short-range repulsion + bookkeeping
    constants" cleanly.

    Parameters
    ----------
    charges
        :class:`SmearedNuclearCharges` bundle.
    system
        :class:`PeriodicSystem` carrying the lattice (used for the
        ``image_shells`` summation). The atoms attribute is *not*
        consumed -- positions come from ``charges``.
    image_shells
        Number of neighbouring images along each axis. For a
        typical cell + a, ``image_shells = 1`` is far more than
        enough; ``erf`` saturates to 1 at long range, and the
        ``1/r`` decay kicks in.
    """
    if image_shells < 0:
        raise ValueError(
            f"image_shells must be >= 0, got {image_shells!r}"
        )
    L = np.asarray(system.lattice, dtype=float)
    n_atoms = charges.n_atoms
    if n_atoms == 0:
        return 0.0

    total = 0.0
    for ia in range(n_atoms):
        Za, alpha_a, Ra = (
            float(charges.Z[ia]),
            float(charges.alpha[ia]),
            charges.positions[ia],
        )
        for ib in range(n_atoms):
            Zb, alpha_b, Rb0 = (
                float(charges.Z[ib]),
                float(charges.alpha[ib]),
                charges.positions[ib],
            )
            beta = (alpha_a * alpha_b) / math.sqrt(
                alpha_a ** 2 + alpha_b ** 2
            )
            for ix in range(-image_shells, image_shells + 1):
                for iy in range(-image_shells, image_shells + 1):
                    for iz in range(-image_shells, image_shells + 1):
                        if ia == ib and ix == 0 and iy == 0 and iz == 0:
                            # Same atom in home cell -- that's the
                            # self-Hartree handled by
                            # smeared_self_energy. Skip.
                            continue
                        shift = (
                            ix * L[:, 0] + iy * L[:, 1] + iz * L[:, 2]
                        )
                        r_vec = Rb0 + shift - Ra
                        r_ab = float(np.linalg.norm(r_vec))
                        if r_ab < 1e-12:
                            # Coincident atoms in different cells --
                            # treat as r->0 smeared-Coulomb limit.
                            total += 0.5 * Za * Zb * (
                                2.0 * beta / math.sqrt(math.pi)
                            )
                        else:
                            total += 0.5 * Za * Zb * (
                                math.erf(beta * r_ab) / r_ab
                            )
    return float(total)


# ============================================================
# Smeared V_ne long-range matrix
# ============================================================


def smeared_v_ne_long_matrix(
    basis,
    charges: SmearedNuclearCharges,
    grid: PlaneWaveGrid,
    *,
    image_shells: int = 1,
    quiet: bool = False,
) -> np.ndarray:
    """Build the **long-range** electron-nucleus attraction matrix
    via FFT-Poisson on the smeared nuclear density.

    Algorithm:

    1. Collocate ``r_core`` on ``grid`` (this module's
       :func:`smeared_nuclear_density_on_grid`).
    2. Solve Poisson with the standard FFT kernel
       (``_vibeqc_core.solve_poisson_coulomb``) -- pins ``Ṽ(G=0)``
       to zero, same gauge as the electronic Hartree-J.
    3. Project the resulting potential onto the AO basis with a
       negative sign (electron-nucleus attraction is negative):

           ``V_ne_long[muν] = - ∫ chi_mu(r) chi_ν(r) V_core_long(r) dr``

    The result is the **smooth-grid** part of V_ne. The
    short-range complement (``- S_a Z_a . erfc(a_a . r) / r``,
    summed over image cells and projected onto chi_mu chi_ν) is **not
    included here** -- it lives on the per-atom radial grid in the
    M3b GAPW path, or via a libint erfc-attenuated nuclear-
    attraction one-electron operator in a GPW-pseudo path. Adding
    those is the load-bearing M3b work.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`.
    charges
        :class:`SmearedNuclearCharges` -- must use the same atomic
        positions and Z values as the basis was built on.
    grid
        :class:`PlaneWaveGrid`.
    image_shells
        Passed through to the density-on-grid summation.
    quiet
        Suppress the experimental-flag warning.

    Returns
    -------
    V_ne_long
        ``(n_basis, n_basis)`` symmetric matrix (Hartree).

    Notes
    -----
    Gauge: this V_ne_long is built in the **same gauge as the
    FFT-Poisson Hartree** (G = 0 dropped). Pairing it with the M2
    GPW J in an SCF would close the V_ne-vs-J gauge mismatch *if*
    the short-range erfc piece were also added -- but on its own,
    smeared-V_ne_long alone is **not** a physically complete V_ne.
    The function exists so M3b has the long-range piece factored
    out cleanly; the M3a Ewald-lattice V_ne (in
    ``periodic_gapw_j._ewald_v_ne_gamma``) remains the production
    path for the M2-full SCF.
    """
    if not quiet:
        warnings.warn(
            f"smeared_v_ne_long_matrix on "
            f"{grid.nx}x{grid.ny}x{grid.nz} grid: this is the "
            f"smooth-grid (erf) piece of V_ne only; the short-"
            f"range erfc complement is M3b work and is NOT "
            f"included in the returned matrix.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )
    rho_core = smeared_nuclear_density_on_grid(
        charges, grid, image_shells=image_shells,
    )
    V_core_long = _core.solve_poisson_coulomb(rho_core, grid.lattice_bohr)
    # Reuse the GPW projector primitive to avoid duplicating the
    # <chi_mu | V | chi_ν> quadrature.
    from .periodic_gapw_j import project_potential_to_ao
    V_ao = project_potential_to_ao(basis, V_core_long, grid)
    # Minus sign: electron-nucleus attraction is negative-valued.
    return -V_ao


def smeared_v_ne_erfc_short_matrix(
    basis,
    system,
    alpha: float,
    *,
    lat_opts=None,
) -> np.ndarray:
    """Build the **short-range** electron-nucleus attraction matrix
    at Γ via libint's erfc-attenuated nuclear-attraction operator.

    Concretely: ``V_ne_short[muν] = <chi_mu | -S_a Z_a . erfc(a . r_a) /
    r_a | chi_ν>`` summed over the lattice cells within
    ``lat_opts.nuclear_cutoff_bohr``, then Bloch-summed at Γ.

    The kernel is ``erfc(a r)/r``, which decays inside ``~3/a`` bohr,
    so the lattice cutoff used by the default ``LatticeSumOptions``
    (``nuclear_cutoff_bohr = 25``) captures the full short-range
    tail to ~``erfc(75) ≈ 1e-26`` for ``a = 1`` bohr⁻¹. For tighter
    ``a``, the cutoff is even more generous.

    Pair with :func:`smeared_v_ne_long_matrix` (which gives the
    long-range smooth-grid piece via FFT-Poisson on r_core). The
    two together reconstruct the full V_ne in the FFT-Poisson
    gauge of the electronic Hartree-J. CP2K-GAPW V_ne convention,
    audit D5(b).

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`.
    system
        :class:`PeriodicSystem`.
    alpha
        Smearing exponent (bohr⁻¹). **Must equal the a used for
        the smeared r_core in** :func:`smeared_v_ne_long_matrix`
        for the long + short reconstruction to cancel into the
        bare 1/r nuclear attraction.
    lat_opts
        :class:`LatticeSumOptions`. Defaults to standard cutoffs
        (15 / 25 bohr).

    Returns
    -------
    V_ne_short
        ``(n_basis, n_basis)`` symmetric, negative-diagonal
        (attractive) matrix.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if lat_opts is None:
        lat_opts = _core.LatticeSumOptions()
    V_lat = _core.compute_nuclear_erfc_lattice(basis, system, alpha, lat_opts)
    V_k = _core.bloch_sum(V_lat, np.zeros(3))
    V = np.real(V_k)
    return 0.5 * (V + V.T)


def smeared_v_ne_full_gamma(
    basis,
    system,
    *,
    alpha: Optional[float] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    lat_opts=None,
    image_shells: int = 1,
    quiet: bool = False,
) -> np.ndarray:
    """Build the **full** Γ-point V_ne in the CP2K-style erfc/erf
    split convention.

    Reconstructs the bare ``- S_a Z_a / r_a`` periodic V_ne as

        ``V_ne = V_ne_long_smeared  +  V_ne_short_erfc``

    where the long-range smooth-grid piece is FFT-Poisson on the
    Gaussian-smeared nuclear density ``r_core(r) = S_a Z_a .
    (a/√pi)^3 exp(-a^2 (r - R_a)^2)``, and the short-range piece is
    the libint ``erfc(a . r)/r`` lattice integral. Both use the
    same ``a`` so the algebraic cancellation that makes the GAPW
    augmentation tractable (CP2K's
    ``qs_core_energies.calculate_ecore_overlap`` convention) is
    honoured by construction.

    This is the audit D5(b) path: the CP2K-style V_ne convention
    that lets every Hartree-class quantity (V_ne_long,
    electronic Hartree J, smeared E_nn) route through the *same*
    FFT-Poisson kernel and share the same v_bg jellium gauge.
    The D5(a) path is the M3a Ewald lattice V_ne -- both are
    physically equivalent on neutral cells; this one is the
    natural fit for the M3b GAPW augmentation.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet`.
    system
        :class:`PeriodicSystem`.
    alpha
        Smearing exponent (bohr⁻¹). Defaults to
        :func:`default_smearing_alpha_from_grid(grid)`.
    grid
        :class:`PlaneWaveGrid`. Pass ``None`` to let the helper
        build one from ``cutoff_ha``.
    cutoff_ha
        PW kinetic-energy cutoff (Ha) for the grid auto-build.
        Default 300 Ha (the M2 demo cutoff). Ignored when
        ``grid`` is supplied.
    lat_opts
        :class:`LatticeSumOptions` for the erfc short-range
        lattice sum. Defaults to standard cutoffs.
    image_shells
        Image-shell count for the smeared-density grid
        collocation.
    quiet
        Suppress the experimental warning.

    Returns
    -------
    V_ne
        ``(n_basis, n_basis)`` symmetric real matrix in Hartree.
        Negative-diagonal on atoms with nonzero Z (attractive).
    """
    if not quiet:
        warnings.warn(
            "smeared_v_ne_full_gamma: M3b experimental V_ne "
            "convention (CP2K erfc/erf split). The M3a Ewald "
            "lattice V_ne in periodic_gapw_j._ewald_v_ne_gamma "
            "is the production-tested path; this one is wired "
            "up for the M3b GAPW augmentation work.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid
        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float),
                           cutoff_ha=cutoff_ha)
    if alpha is None:
        alpha = default_smearing_alpha_from_grid(grid)
    charges = SmearedNuclearCharges.from_periodic_system_uniform_alpha(
        system, alpha=alpha,
    )
    V_long = smeared_v_ne_long_matrix(
        basis, charges, grid, image_shells=image_shells, quiet=True,
    )
    V_short = smeared_v_ne_erfc_short_matrix(
        basis, system, alpha, lat_opts=lat_opts,
    )
    # The "a-correction" (standard PW-DFT G=0 bookkeeping). FFT-Poisson
    # on the smeared nuclear density r_core drops the G=0 reciprocal
    # mode, effectively coupling r_core to a uniform compensating
    # negative-charge background; the smeared V_ne therefore lives in
    # a gauge that differs from V_ne_bare by a constant `c_a`. To
    # close the gauge so SCF totals match the M3a Ewald path on a
    # neutral cell, we add ``c_a . S`` with
    #
    #     c_a = pi . (S_a Z_a) / (a^2 . V_cell)
    #
    # giving a total-energy contribution
    # ``tr(D . c_a . S) = c_a . N_elec = pi . (S Z)^2 / (a^2 . V_cell)``
    # for a neutral cell. This is the same constant CP2K names
    # ``E_self_a`` / ``alpha_madelung`` in qs_core_energies.F; it
    # falls out as the G = 0 limit of the smeared-charge / uniform-bg
    # interaction, derived directly from the FFT-Poisson convention.
    # Empirically pinned to < 1 µHa on He and H2 STO-3G across
    # ``a in {0.5, 1.0, 1.5, 2.0, 2.5}`` -- the formula closes the
    # full convention bookkeeping for uniform-a bundles.
    from .periodic_gapw_j import _overlap_lattice_gamma
    Z_total = float(charges.Z.sum())
    V_cell = grid.cell_volume_bohr3
    c_alpha = math.pi * Z_total / (alpha ** 2 * V_cell)
    S = _overlap_lattice_gamma(basis, system)
    return V_long + V_short + c_alpha * S


# ============================================================
# Multipole-compensating Gaussian density (r_0)
# ============================================================
#
# The r_0 compensator (CP2K's `qs_rho0_methods.calculate_rho0_atom`)
# is the GAPW workhorse for fixing long-range multipole errors in the
# soft FFT density. Each atom carries a Gaussian-shaped density whose
# moments Q_lm reproduce a target multipole pattern (typically the
# hard-minus-soft atomic density inside the augmentation sphere).
#
# The Gaussian-times-real-solid-harmonic form CP2K uses, paraphrased:
#
#   r_0(r) = S_a S_lm Q_lm,a . N_l(a) . r_a^l . S_lm(Ω̂_a) .
#              . exp(-a^2 r_a^2)
#
# where ``r_a, Ω̂_a`` are radial and angular coords relative to atom a,
# and ``N_l(a)`` normalises so that ``∫ r_0(r) . r_a^l . S_lm(Ω̂_a) dr
# = Q_lm,a``. With Stone-normalised real spherical harmonics,
#
#     ``N_l(a) = 2^(l + 2) . a^(2l + 3) / Γ(l + 3/2) / √pi``
#
# (a standard result; see Stone, *Theory of Intermolecular Forces*,
# Appendix C). For l = 0 this reduces to the smeared-charge
# normalisation ``(a/√pi)^3`` used by SmearedNuclearCharges -- the M3a
# / M3b smeared nuclear density is just the monopole (l = 0)
# component of this general compensator with Q_00 = Z . √(4pi).
#
# We ship the l = 0 + l = 1 dipole terms here; higher l can be added
# by extending the closed-form table. M3b GAPW augmentation will
# extend through l = lmax (typically 2-4 depending on basis).


def _multipole_normalisation(l: int, alpha: float) -> float:
    """Return ``K_l(a) = 2 . a^(2l + 3) / Γ(l + 3/2)``.

    Chosen so that the Gaussian-multipole density

        ``g_lm(r) = K_l(a) . r^l . S_lm(Ω̂_r) . exp(-a^2 r^2)``

    integrated against ``r^l . S_l'm'(Ω̂_r)`` over 3-space returns
    ``d_ll' . d_mm'``: i.e., ``K_l`` makes the Gaussian carry unit
    ``Q_lm`` per unit moment coefficient. Sanity:

    * ``l = 0``: ``K_0 = 4 a^3 / √pi``, and r(r) = q . K_0 . S_00 .
      exp(-a^2 r^2) = q . (a/√pi)^3 exp(-a^2 r^2) when ``q = Q_00 . √(4pi)``
      via the conversion in :func:`monopole_compensator_for_charge`.
    * ``l = 1``: ``K_1 = 2 a⁵``, the standard p-Gaussian
      normalisation factor.
    """
    if l < 0:
        raise ValueError(f"l must be >= 0; got {l!r}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive; got {alpha!r}")
    return 2.0 * alpha ** (2 * l + 3) / math.gamma(l + 1.5)


@dataclass(frozen=True)
class GaussianMultipoleCompensator:
    """Per-atom Gaussian-shaped multipole-compensating density (r_0).

    Stores per-atom positions, Gaussian widths a, and multipole
    moments ``Q_lm`` (flat-indexed by
    :func:`vibeqc.periodic_gapw_atomic_grid.multipole_index`).
    Method :meth:`density_on_grid` lays the compensator down on an
    FFT grid; the result has the property that its multipole moments
    ``Q_lm,a`` per atom equal the supplied ``Q_lm`` values.

    Currently supports l = 0 (monopole) and l = 1 (dipole). Higher l
    requires extending the hard-coded ``S_lm`` table in the atomic-
    grid module; see :func:`vibeqc.periodic_gapw_atomic_grid.real_spherical_harmonics`.

    Attributes
    ----------
    positions
        ``(n_atoms, 3)`` Cartesian positions in bohr.
    alpha
        ``(n_atoms,)`` Gaussian widths (bohr⁻¹).
    Q
        ``(n_atoms, n_components)`` per-atom multipole moments,
        where ``n_components = (lmax + 1)^2``.
    lmax
        Maximum angular momentum represented. ``Q`` must have
        ``(lmax + 1)^2`` columns.
    """

    positions: np.ndarray
    alpha: np.ndarray
    Q: np.ndarray
    lmax: int

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        alpha = np.asarray(self.alpha, dtype=float)
        Q = np.asarray(self.Q, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (n_atoms, 3); got {positions.shape}"
            )
        n_atoms = positions.shape[0]
        if alpha.shape != (n_atoms,):
            raise ValueError(
                f"alpha shape {alpha.shape} mismatches positions "
                f"({n_atoms} atoms)"
            )
        if not np.all(alpha > 0.0):
            raise ValueError("alpha entries must be positive")
        if self.lmax < 0:
            raise ValueError(
                f"GaussianMultipoleCompensator needs lmax >= 0; got "
                f"lmax={self.lmax!r}"
            )
        n_comp = (self.lmax + 1) ** 2
        if Q.shape != (n_atoms, n_comp):
            raise ValueError(
                f"Q shape {Q.shape} must be (n_atoms={n_atoms}, "
                f"n_components={n_comp}) for lmax={self.lmax}"
            )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "Q", Q)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    def density_on_grid(
        self,
        grid: PlaneWaveGrid,
        *,
        image_shells: int = 1,
    ) -> np.ndarray:
        """Lay the multipole-compensating density down on an FFT grid.

        For each atom ``a`` and each (l, m) component, contribute
        ``Q_lm,a . N_l(a_a) . r_a^l . S_lm(Ω̂_a) . exp(-a_a^2 r_a^2)``
        to the grid. The result integrates to ``S_a Q_00,a / √(4pi)``
        in total charge (Stone-normalised) and has the per-atom
        multipoles encoded in :attr:`Q`.

        Periodic image shells are summed to capture the Gaussian
        tails that cross the cell boundary, identical pattern to
        :func:`collocate_point_charges_on_grid`.
        """
        # Cartesian grid points in absolute bohr coords.
        from .periodic_gapw_atomic_grid import (
            real_spherical_harmonics, multipole_index,
        )
        r_xyz = grid.cartesian_coords()  # (nx, ny, nz, 3)
        L = grid.lattice_bohr            # columns a1, a2, a3
        rho = np.zeros(grid.shape, dtype=float)
        for ia in range(self.n_atoms):
            R_a = self.positions[ia]
            alpha = float(self.alpha[ia])
            Q_a = self.Q[ia]
            for ix in range(-image_shells, image_shells + 1):
                for iy in range(-image_shells, image_shells + 1):
                    for iz in range(-image_shells, image_shells + 1):
                        shift = (
                            ix * L[:, 0] + iy * L[:, 1] + iz * L[:, 2]
                        )
                        centre = R_a + shift
                        # (nx, ny, nz, 3) displacement vectors.
                        delta = r_xyz - centre
                        r2 = (delta * delta).sum(axis=-1)
                        r = np.sqrt(r2)
                        gauss = np.exp(-alpha ** 2 * r2)
                        # Build the unit-vector ̂Ω. Guard against r=0
                        # at the centre -- there S_lm is undefined for
                        # l>0; we use a small e floor.
                        r_safe = np.where(r > 1e-30, r, 1.0)
                        omega = delta / r_safe[..., None]
                        flat_omega = omega.reshape(-1, 3)
                        S = real_spherical_harmonics(
                            flat_omega, self.lmax,
                        )
                        # S has shape ((lmax+1)^2, n_grid).
                        for l in range(self.lmax + 1):
                            Nl = _multipole_normalisation(l, alpha)
                            r_l = r ** l
                            for m in range(-l, l + 1):
                                idx = multipole_index(l, m)
                                S_lm = S[idx].reshape(grid.shape)
                                rho += (
                                    Q_a[idx] * Nl * r_l * S_lm * gauss
                                )
        return rho


def monopole_compensator_for_charge(
    positions_bohr: np.ndarray,
    Z: np.ndarray,
    alpha: float,
) -> GaussianMultipoleCompensator:
    """Build a pure-monopole (lmax = 0) compensator carrying nuclear
    charges ``Z`` on the given positions, with the convention that
    the integrated density of each atom equals ``Z_a``.

    Concretely sets ``Q_00,a = Z_a . √(4pi)`` so that the
    Stone-normalised compensator integrates to ``Z_a`` per atom
    (matching the
    :func:`vibeqc.periodic_gapw_smearing.smeared_nuclear_density_on_grid`
    convention).
    """
    positions = np.asarray(positions_bohr, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions_bohr must be (n_atoms, 3); got {positions.shape}"
        )
    if Z_arr.shape != (positions.shape[0],):
        raise ValueError(
            f"Z shape {Z_arr.shape} doesn't match positions "
            f"({positions.shape[0]} atoms)"
        )
    n_atoms = positions.shape[0]
    # Stone normalisation: Q_00 = ∫r . S_00 d^3r = (1/√(4pi)) . ∫r d^3r,
    # so for a Gaussian carrying total charge Z, Q_00 = Z / √(4pi).
    Q = (Z_arr / math.sqrt(4.0 * math.pi)).reshape(n_atoms, 1)
    return GaussianMultipoleCompensator(
        positions=positions,
        alpha=np.full(n_atoms, float(alpha)),
        Q=Q,
        lmax=0,
    )


# ============================================================
# Electronic r_0: per-atom multipole compensator for polar cells
# ============================================================


def compute_atom_local_multipoles(
    density_matrix: np.ndarray,
    basis,
    system,
    atom_idx: int,
    *,
    lmax: int = 2,
    n_radial: int = 50,
    lebedev_order: int = 17,
    radial_alpha: Optional[float] = None,
) -> np.ndarray:
    """Compute multipole moments ``Q_lm`` of the electronic density
    on an augmentation sphere around one atom (M3-aug).

    Builds a per-atom radial x Lebedev grid centred on
    ``system.unit_cell[atom_idx]``, evaluates ``r(r) = S_muν D_muν
    chi_mu(r) chi_ν(r)`` on those grid points, and computes the
    standard Stone-normalised multipole moments via
    :func:`vibeqc.periodic_gapw_atomic_grid.compute_multipole_moments`.

    Useful for diagnostic checks of polar-cell behaviour and as
    the building block for a self-consistent r_0 electronic
    multipole compensator (future M3+ work).

    Parameters
    ----------
    density_matrix
        ``(n_basis, n_basis)`` AO density.
    basis, system
        Standard vibe-qc periodic system + basis.
    atom_idx
        Which atom of ``system.unit_cell`` to centre the grid on.
    lmax, n_radial, lebedev_order
        :class:`AtomicRadialGrid` knobs (defaults: 50 radial x 17
        Lebedev, ``lmax = 2`` to cover monopole + dipole +
        quadrupole).
    radial_alpha
        Mura-Knowles a for the radial grid (default per-element
        via :func:`vibeqc.periodic_gapw_atomic_grid.default_alpha_for_element`).

    Returns
    -------
    Q
        ``((lmax + 1)^2,)`` array of multipole moments.
    """
    from . import _vibeqc_core as core
    from .periodic_gapw_atomic_grid import (
        AtomicRadialGrid, compute_multipole_moments,
        default_alpha_for_element,
    )
    if atom_idx < 0 or atom_idx >= len(system.unit_cell):
        raise ValueError(
            f"atom_idx out of range; got {atom_idx} for "
            f"{len(system.unit_cell)} atoms"
        )
    atom = system.unit_cell[atom_idx]
    centre = np.asarray(list(atom.xyz), dtype=float)
    if radial_alpha is None:
        radial_alpha = default_alpha_for_element(int(atom.Z))
    grid = AtomicRadialGrid.build(
        centre, n_radial=n_radial, alpha=radial_alpha,
        lebedev_order=lebedev_order,
    )
    pts = grid.cartesian_points().reshape(-1, 3)
    chi = np.asarray(core.evaluate_ao(basis, pts))  # (n_grid, n_basis)
    D = np.asarray(density_matrix, dtype=float)
    rho = np.einsum("gm,mn,gn->g", chi, D, chi, optimize=True)
    rho = rho.reshape(grid.n_radial, grid.n_angular)
    return compute_multipole_moments(rho, grid, lmax)


def build_electronic_compensator(
    density_matrix: np.ndarray,
    basis,
    system,
    *,
    lmax: int = 2,
    smearing_alpha: float = 1.5,
    n_radial: int = 50,
    lebedev_order: int = 17,
) -> "GaussianMultipoleCompensator":
    """Build a multipole-matched r_0 compensator for the electronic
    density on a polar cell.

    The compensator carries, per atom, the same multipoles
    ``Q_lm`` as the converged electronic density inside that
    atom's augmentation sphere. Adding it to the FFT density
    fixes the long-range multipole leakage that the FFT-Poisson
    G = 0 = 0 gauge introduces on polar cells (the higher-l
    cousin of the monopole "alpha-correction" already applied to
    the smeared nuclear density).

    Currently this builds the compensator as a **diagnostic** --
    not yet wired into the SCF Hartree-J path. A self-consistent
    integration requires also subtracting the compensator's
    analytic in-sphere contribution from the Hartree matrix
    elements (the "hard" side of the dual-grid bookkeeping),
    which is future M3+ work.

    Parameters
    ----------
    density_matrix, basis, system
        Standard vibe-qc periodic system + basis + AO density.
    lmax
        Multipole order. Default 2 (monopole + dipole +
        quadrupole).
    smearing_alpha
        Gaussian width a (bohr⁻¹) for the compensator's per-atom
        Gaussians. Default 1.5 -- wide enough that the
        compensator is smooth on the FFT grid; tight enough that
        atoms don't overlap.
    n_radial, lebedev_order
        Knobs forwarded to :func:`compute_atom_local_multipoles`.
    """
    n_atoms = len(system.unit_cell)
    positions = np.array(
        [list(a.xyz) for a in system.unit_cell], dtype=float
    )
    n_comp = (lmax + 1) ** 2
    Q = np.zeros((n_atoms, n_comp), dtype=float)
    for ia in range(n_atoms):
        Q[ia] = compute_atom_local_multipoles(
            density_matrix, basis, system, ia,
            lmax=lmax, n_radial=n_radial,
            lebedev_order=lebedev_order,
        )
    return GaussianMultipoleCompensator(
        positions=positions,
        alpha=np.full(n_atoms, float(smearing_alpha)),
        Q=Q,
        lmax=lmax,
    )
