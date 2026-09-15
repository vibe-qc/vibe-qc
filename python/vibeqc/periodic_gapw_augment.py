"""GAPW augmentation -- per-atom hard/soft Hartree and XC decomposition.

> **Experimental.** This module is the M3c GAPW augmentation milestone
> (``docs/design_periodic_gapw.md`` Sec. M3). It wires the per-atom radial-
> grid infrastructure (M3b) into a Hartree and XC correction that turns
> the smooth-grid GPW J build into an all-electron GAPW J build.
>
> The GAPW route is **not yet the default** -- ``PeriodicJKMethod.AUTO``
> stays on GDF/BIPOLE. The user opts in via ``jk_method="gapw"`` on
> ``run_periodic_job``.

GAPW (Lippert, Hutter & Parrinello, *Theor. Chem. Acc.* 103, 124 (1999))
decomposes the total-energy Hartree and XC contributions as:

**Hartree decomposition:**

    ``V_H[r] = V_H[r̃ + r₀] + S_a (V_H[r_a] - V_H[r̃_a + r₀,a])``

where:
- r̃ is the **smooth** density on the FFT grid (same as GPW),
- r₀ is the **multipole compensator** (Gaussian density matching the
  hard-minus-soft multipoles),
- r_a is the **all-electron** atomic density inside the augmentation
  sphere around atom a,
- r̃_a is the **soft** (pseudo) atomic density,
- r₀,a is the atomic part of the multipole compensator.

The first term is the smooth FFT-Poisson solve (identical to GPW but
on the compensated density r̃ + r₀). The sum over atoms corrects for
the difference between all-electron and pseudo behaviour inside each
augmentation sphere, computed on per-atom radial x Lebedev grids.

**XC decomposition:**

    ``E_xc[r] = E_xc[r̃] + S_a (E_xc[r_a] - E_xc[r̃_a])``

The same additive correction pattern: evaluate XC on the smooth grid
for r̃, then add per-atom corrections on the radial grids.

**Implementation strategy for Gaussian-basis GAPW**

In a Gaussian-orbital code without PAW datasets, the "all-electron"
atomic density r_a and "pseudo" atomic density r̃_a are modelled from
the Gaussian basis:

1. **r_a(r)** -- the AO density evaluated on the per-atom radial grid
   using the unmodified basis. Inside a small augmentation sphere
   (radius r_c ~ 2-4 bohr) this captures the hard nodal structure.

2. **r̃_a(r)** -- a "softened" atomic density built by expanding the
   AO density in a basis of softened (narrower exponent range)
   Gaussians, or equivalently by applying a cutoff filter to the
   original AOs so that tight primitives (exponent > a_cut) are
   dropped inside the augmentation sphere.

3. The augmentation correction subtracts the soft contribution
   and adds back the hard contribution pointwise on the radial
   grid before projecting to the AO basis, so the final J and V_xc
   matrices are all-electron correct.

For the initial production implementation we use a direct Gaussian-
product model: the all-electron AO density is evaluated on the atomic
grids, and the soft density is modelled by removing the tightest
primitives. This is equivalent to what CP2K's GAPW does with a
"basis set truncation" type of pseudisation.

**Dual-grid Hartree solve on the atomic sphere**

For each atom a:

1. Build the radial x Lebedev grid.
2. Evaluate chi_mu(r) at each grid point.
3. Form r_a(r) = S_muν D_muν . chi_mu(r) . chi_ν(r).
4. Build r̃_a(r) similarly from the softened basis.
5. Form r₀,a(r) analytically from the Gaussian multipole compensator.
6. Solve Poisson on the radial grid for V_H[r_a] and V_H[r̃_a + r₀,a].
7. Compute the correction to the AO J matrix:
   ``ΔJ_muν = ∫_{sphere} chi_mu(r) chi_ν(r) [V_H[r_a](r) - V_H[r̃_a + r₀,a](r)] dr``
8. The correction ΔJ is added to the smooth FFT-Poisson J.

The same pattern applies for the XC correction:
   ``ΔV_xc,muν = ∫_{sphere} chi_mu(r) chi_ν(r) [v_xc[r_a](r) - v_xc[r̃_a](r)] dr``
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from . import _vibeqc_core as _core
from .periodic_screened_exchange import reject_unscreened_range_separated
from .periodic_gapw_atomic_grid import (
    AtomicRadialGrid,
    GAPWExperimentalWarning,
    compute_multipole_moments,
    default_alpha_for_element,
    multipole_index,
    n_multipole_components,
)
from .periodic_gapw_grid import PlaneWaveGrid
from .periodic_gapw_j import (
    _COMPACT_GPW_LINDEP_THRESHOLD,
    GpwCollocationCache,
    GpwEnergyBreakdown,
    GpwJBuilder,
    GpwMultiKScfResult,
    GpwScfResult,
    _ao_values_on_grid,
    _build_v_ne,
    _compute_d3_correction,
    _eigh_safe,
    _evaluate_xc_on_grid,
    _kinetic_lattice_gamma,
    _make_diis,
    _multik_gpw_is_molecular_limit,
    _normalise_initial_density_k,
    _overlap_lattice_gamma,
    _project_vtau_to_ao,
    _project_vxc_to_ao,
    _weighted_real_density_from_k,
    _xc_effective_potential_grid,
    bloch_ao_on_grid,
    build_gpw_collocation_cache,
    collocate_bloch_density_on_grid,
    collocate_density_on_grid,
    compute_bloch_kinetic_energy_density,
    project_potential_to_ao,
    project_potential_to_bloch_ao,
    project_vtau_to_bloch_ao,
)
from .periodic_gapw_smearing import (
    GaussianMultipoleCompensator,
    default_smearing_alpha_from_grid,
)
from .smearing.apply import _global_aufbau_with_mu
from .smearing import (
    SmearingOptions as _SmearingOptions,
)
from .smearing import (
    apply_smearing as _apply_smearing,
)

__all__ = [
    "GapwAugmentation",
    "GapwJBuilder",
    "GapwScfResult",
    "GapwUhfScfResult",
    "GapwUksScfResult",
    "run_periodic_rhf_gapw",
    "run_periodic_rks_gapw",
    "run_periodic_rks_gapw_multi_k",
    "run_periodic_uhf_gapw",
    "run_periodic_uks_gapw",
    "softened_basis",
    "solve_poisson_radial",
    "estimate_augmentation_radius",
]


# ============================================================
# Constants and defaults
# ============================================================

# Default augmentation sphere radius (bohr). Inside this sphere the
# per-atom radial grid correction is applied. CP2K uses per-element
# radii (covalent radius + ~0.5-1.0 bohr); the single default here
# works for light elements.
_DEFAULT_AUG_RADIUS = 3.5  # bohr

# Default multipole order for the r₀ compensator.
# Multi-k GAPW: how much excess charge the Gamma-folded density may
# carry before its Hartree term is refused. The fold double-counts
# inter-cell density whenever AO products reach neighbouring cells;
# measured excess (STO-3G, Hcore guess): He/8-bohr (1,1,2) 0.0 %,
# He/6-bohr (2,2,2) +0.5 %, LiH rocksalt (2,2,2) +145 %. 1 % separates
# the working configurations from the broken one with two orders of
# magnitude to spare, and sits above the grid-collocation quadrature
# error (which under-integrates slightly; see the boundary-atom note in
# docs/user_guide/gapw.md).
_MULTIK_FOLD_CHARGE_TOL = 0.01

_DEFAULT_LMAX = 3  # s, p, d, f

# Default number of radial points for the atomic augmentation grid.
_DEFAULT_N_RADIAL = 80

# Default Lebedev order.
_DEFAULT_LEBEDEV_ORDER = 17


def _von_weizsaecker_floor(
    tau: np.ndarray, rho: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Clamp t to its von Weizsacker lower bound t >= sigma / (8 rho).

    The exact single-orbital value t_W bounds the true kinetic-energy
    density from below; a grid-built t can dip below it from
    finite-difference / spectral roundoff, where unregularised meta-GGAs
    (TPSS, M06-L) are singular. Applied identically to the hard and soft
    atomic t so the augmentation telescoping stays consistent. Mirrors
    ``enforce_von_weizsaecker_floor`` in cpp/src/periodic_xc.cpp and the
    smooth-grid floor in periodic_gapw_j._evaluate_xc_on_grid. t is set to
    0 where rho <= 0.
    """
    tau_w = np.where(rho > 0.0, sigma / (8.0 * rho), 0.0)
    return np.where(rho > 0.0, np.maximum(tau, tau_w), 0.0)

# Primitive exponent cutoff for the "soft" density (bohr⁻^2).
# Primitives with exponent > a_soft_cutoff are dropped from the
# softened basis, giving a smooth density that misses the hard
# nodal structure near the nucleus.
_DEFAULT_SOFT_CUTOFF = 3.0  # bohr⁻^2 (for Gaussian exponents)


def estimate_augmentation_radius(Z: int) -> float:
    """Estimate an augmentation sphere radius for element Z.

    Returns a radius (bohr) inside which the GAPW hard/soft
    correction is applied. Uses covalent radius + 1.0 bohr padding,
    capped at a minimum of 2.0 bohr.

    Parameters
    ----------
    Z
        Atomic number.

    Returns
    -------
    radius
        Augmentation sphere radius in bohr.
    """
    # Covalent radii (bohr): H = 0.6, C = 1.4, N = 1.3, O = 1.2,
    # F = 1.1, He-Ne: interpolate. Fallback for heavier elements.
    _cov_radii = {
        1: 0.6,
        2: 0.6,
        3: 2.6,
        4: 2.0,
        5: 1.6,
        6: 1.4,
        7: 1.3,
        8: 1.2,
        9: 1.1,
        10: 1.0,
    }
    cov = _cov_radii.get(Z, 1.0 + 0.1 * max(0, Z - 10))
    return max(2.0, cov + 1.0)


# ============================================================
# Softened basis construction
# ============================================================


def softened_basis(
    basis,
    mol_or_system,
    *,
    soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
) -> object:
    """Build a "softened" basis by removing tight primitives.

    Uses the C++ ``prune_tight_primitives`` kernel which operates
    on the native :class:`BasisSet` internals. Falls back to a
    no-op Python path if the C++ binding is unavailable.

    Parameters
    ----------
    basis
        Original :class:`vibeqc.BasisSet`.
    mol_or_system
        :class:`vibeqc.Molecule` or :class:`PeriodicSystem`.
        The molecule is extracted and passed to the C++ kernel
        so it can build the new :class:`BasisSet` from pruned
        shells.
    soft_cutoff
        Exponent threshold (bohr⁻^2). Primitives with exponent >
        this value are removed from each contracted shell on core-bearing
        atoms. Hydrogen has no core, so its contractions are retained in
        full even when one primitive exceeds the numerical threshold.

    Returns
    -------
    soft_basis
        A new :class:`BasisSet` with only the soft primitives.
    """
    # Resolve the molecule. The multiplicity is inferred from the
    # total electron count (odd n_elec -> doublet, even -> singlet)
    # so that the Molecule constructor is satisfied.
    from .molecule import Molecule

    if hasattr(mol_or_system, "unit_cell"):
        n_elec = int(sum(int(a.Z) for a in mol_or_system.unit_cell))
        mult = 2 if n_elec % 2 != 0 else 1
        mol = Molecule(list(mol_or_system.unit_cell), 0, mult)
    else:
        mol = mol_or_system

    # A scalar exponent cutoff is only a useful core/valence proxy on atoms
    # that actually have a core. Hydrogen's sole STO-3G contraction contains
    # a 3.425 bohr^-2 primitive, just above the historical 3.0 cutoff. Pruning
    # it changes the bonding AO rather than separating a hard core density and
    # causes a grid-independent ~108 mHa H2 GAPW-HF residual. Preserve H shells
    # exactly. Mixed H/heavy systems take this small construction path so the
    # heavy atoms are still softened; heavy-only systems retain the native fast
    # path below.
    atomic_numbers = [int(atom.Z) for atom in mol.atoms]
    if 1 in atomic_numbers:
        original_shells = basis.shells()
        pruned_shells = []
        changed = False
        for sh in original_shells:
            preserve_contraction = atomic_numbers[int(sh.atom_index)] == 1
            kept_exp = []
            kept_coef = []
            for exponent, coefficient in zip(sh.exponents, sh.coefficients):
                if preserve_contraction or exponent <= soft_cutoff:
                    kept_exp.append(exponent)
                    kept_coef.append(coefficient)
            if not kept_exp:
                changed = True
                continue
            changed = changed or len(kept_exp) != len(sh.exponents)
            pruned = _core.ShellInfo()
            pruned.atom_index = sh.atom_index
            pruned.l = sh.l
            pruned.pure = sh.pure
            pruned.exponents = kept_exp
            pruned.coefficients = kept_coef
            pruned.origin = sh.origin
            pruned_shells.append(pruned)

        if not changed:
            return basis
        return _core.BasisSet(
            mol,
            pruned_shells,
            f"{basis.name}-soft",
            coefficients_pre_normalized=True,
        )

    try:
        return _core.prune_tight_primitives(basis, mol, soft_cutoff)
    except AttributeError:
        # C++ kernel not available -- filter shells in pure Python.
        pass

    # Pure-Python fallback: filter each shell's primitives.
    shells = basis.shells()
    pruned_shells = []
    for sh in shells:
        # Keep only primitives with exponent <= soft_cutoff.
        # If all primitives are removed, skip this shell entirely.
        kept_exp = []
        kept_coef = []
        for e, c in zip(sh.exponents, sh.coefficients):
            if e <= soft_cutoff:
                kept_exp.append(e)
                kept_coef.append(c)
        if not kept_exp:
            continue  # shell has no soft primitives -- drop it
        # Build a new ShellInfo with the pruned primitives.
        # coefficients_pre_normalized=True keeps the contraction
        # coefficients as-is (no libint re-normalisation).
        pruned = _core.ShellInfo()
        pruned.atom_index = sh.atom_index
        pruned.l = sh.l
        pruned.pure = sh.pure
        pruned.exponents = kept_exp
        pruned.coefficients = kept_coef
        pruned.origin = sh.origin
        pruned_shells.append(pruned)

    if not pruned_shells:
        warnings.warn(
            "softened_basis: all primitives pruned -- returning original basis.",
            category=GAPWExperimentalWarning,
        )
        return basis

    return _core.BasisSet(
        mol,
        pruned_shells,
        f"{basis.name}-soft",
        coefficients_pre_normalized=True,
    )


# ============================================================
# Radial Poisson solver
# ============================================================


def solve_poisson_radial(
    density: np.ndarray,
    grid: AtomicRadialGrid,
    *,
    lmax: int = 0,
) -> np.ndarray:
    """Solve Poisson's equation on the per-atom radial x Lebedev grid.

    Multipole-capable radial Poisson solver. The **default ``lmax=0``**
    is the spherical (monopole) solve that the production GAPW
    augmentation uses today; ``lmax > 0`` activates the full multipole
    expansion. The multipole solver itself is correct (it reduces
    bit-for-bit to the monopole for a spherical density), but the GAPW
    *augmentation* is not yet self-consistent at ``l > 0``: the smooth
    r₀ compensator and the boundary windowing do not cancel the hard/soft
    ``l > 0`` pieces inside the sphere, so enabling multipoles in the
    augmentation currently *degrades* molecular totals (an H₂O regression
    was measured). Closing that -- the aspherical augmentation, GPW-AUDIT-
    007 -- is the next milestone; until then the augmentation stays at
    ``lmax=0`` and this routine ships multipole-ready for that work.

    Full **multipole** expansion (Lippert, Hutter & Parrinello, *Theor.
    Chem. Acc.* 103, 124 (1999), Sec. 2; Krack & Parrinello, *Phys. Chem.
    Chem. Phys.* 2, 2105 (2000)). The density is expanded in real
    spherical harmonics on the Lebedev nodes,

        ``r(r, Ω) = S_lm r_lm(r) S_lm(Ω)``,  ``r_lm(r) = ∫ r(r,Ω) S_lm(Ω) dΩ``,

    and each radial component is solved with the multipole Green's
    function of the Coulomb kernel (the 1/|r-r'| expansion in spherical
    harmonics; Jackson, *Classical Electrodynamics*, Eq. 3.70):

        ``V_lm(r) = 4pi/(2l+1) . [ r^-(l+1) ∫₀^r r_lm(r') r'^(l+2) dr'
                                  + r^l    ∫_r^inf r_lm(r') r'^(1-l) dr' ]``,

    reconstructed as ``V(r, Ω) = S_lm V_lm(r) S_lm(Ω)``. The grid weight
    ``w_r`` already carries the r^2 Jacobian, so ``r'^(l+2) dr' ->
    r'^l.w_r`` and ``r'^(1-l) dr' -> r'^-(l+1).w_r``.

    The ``l = 0`` term is the spherical (monopole) solve this routine
    used to do *exclusively*; the ``l > 0`` terms are what GPW-AUDIT-007
    was missing. They matter for the **aspherical** densities of bonded
    atoms (an O in H₂O, an open p shell), where the spherical average
    discards the quadrupole-and-higher Hartree self-interaction inside
    the augmentation sphere. For a genuinely spherical density (closed
    (sub)shell atom) the ``l > 0`` components vanish numerically and the
    result reduces to the monopole solve.

    Parameters
    ----------
    density
        ``(n_radial, n_angular)`` density on the atomic grid.
    grid
        :class:`AtomicRadialGrid` defining the quadrature.
    lmax
        Maximum multipole order. Default ``0`` (monopole; the production
        augmentation setting). Pass ``lmax >= 2`` to capture aspherical
        content (the augmentation-consistency work). The Lebedev order
        must integrate ``S_lm.S_l'm'`` exactly up to ``2.lmax``.

    Returns
    -------
    V_H
        ``(n_radial, n_angular)`` Hartree potential (Hartree) at each
        grid point.
    """
    density = np.asarray(density, dtype=float)
    r = np.asarray(grid.r, dtype=float)
    w_r = np.asarray(grid.w_r, dtype=float)  # carries the r^2 Jacobian
    w_a = np.asarray(grid.w_a, dtype=float)
    nr = grid.n_radial
    na = grid.n_angular
    lmax = max(0, int(lmax))

    # Real spherical harmonics on the Lebedev nodes: (n_comp, n_angular).
    S = grid.evaluate_real_spherical_harmonics(lmax)

    # Angular projection r_lm(r_k) = S_a r(r_k, Ω_a) S_lm(Ω_a) w_a.
    rho_lm = np.einsum("ka,ca,a->ck", density, S, w_a)  # (n_comp, n_radial)

    r_safe = np.where(r > 1e-30, r, 1e-30)
    V_H = np.zeros((nr, na), dtype=float)
    for l in range(lmax + 1):
        r_l = r_safe**l  # r^l
        r_inv_lp1 = r_safe ** (-(l + 1))  # r^-(l+1)
        pref = 4.0 * math.pi / (2 * l + 1)
        for m in range(-l, l + 1):
            c = multipole_index(l, m)
            rho_c = rho_lm[c]
            if not np.any(np.abs(rho_c) > 1e-13):
                continue  # spherical density => no l>0 content
            # Inner A(r_k) = ∫₀^{r_k} r_lm r'^(l+2) dr' = S_{k'<=k} r_lm r^l w_r.
            A = np.cumsum(rho_c * r_l * w_r)
            # Outer B(r_k) = ∫_{r_k}^inf r_lm r'^(1-l) dr' = S_{k'>=k} r_lm r^-(l+1) w_r.
            B = np.cumsum((rho_c * r_inv_lp1 * w_r)[::-1])[::-1]
            # Both cumulative sums are inclusive, so the k' = k shell is
            # counted once in A and once in B. At r' = r the inner and
            # outer kernels coincide (r^-(l+1).r'^l.w = r^l.r'^-(l+1).w =
            # rho.w/r), so the duplicate is exactly one diagonal term and
            # must be subtracted. Without this the self-shell was double-
            # counted: a systematic POSITIVE error ~ rho(r_k).w_k/r_k per
            # point that overestimated every Gaussian Hartree self-energy
            # by 5-13% at n_radial=80 (worse for sharper densities, and
            # only ~halved by doubling n_radial). In the GAPW hard-soft
            # augmentation DIFFERENCE most of it cancels -- which is how
            # it hid from the total-energy parity gates -- but every
            # individual radial Hartree quantity (per-channel energies,
            # the compensator self-energy, any future aspherical l > 0
            # augmentation) carried the full error.
            diag = rho_c * w_r / r_safe
            V_l = pref * (r_inv_lp1 * A + r_l * B - diag)
            V_H += np.outer(V_l, S[c])
    return V_H


# ============================================================
# GAPW augmentation data per atom
# ============================================================


@dataclass
class _AtomAugmentation:
    """Per-atom data for the GAPW augmentation correction.

    Stores the atomic radial grid, the AO values on that grid,
    and the soft (pseudo) AO values.

    Attributes
    ----------
    atom_idx
        Index in the unit cell.
    Z
        Atomic number.
    centre_bohr
        ``(3,)`` Cartesian position (bohr).
    grid
        :class:`AtomicRadialGrid` for this atom.
    aug_radius
        Augmentation sphere radius (bohr).
    chi_a
        ``(n_grid, n_basis)`` AO values on the atomic grid
        (original, "hard" basis). ALL AOs -- used for the ΔJ / ΔV_xc
        *projection* (the Fock is full-dimensional).
    chi_tilde
        ``(n_grid, n_basis_soft)`` AO values on the atomic grid
        (softened basis), or ``None`` if the basis is unfiltered.
    on_center_full
        ``(n_basis,)`` bool mask selecting the AOs centred on THIS
        atom (full basis). The one-center density n_A¹ that sources
        the augmentation potential is built from on-centre AOs only
        (Lippert-Hutter-Parrinello 1999, Eq 53; off-centre tails are
        marginal, p.131). Crucially this keeps a neighbour's AO -- large
        on A's far grid points and amplified by the r^l multipole weight
        -- out of n_A¹, which otherwise blows the compensator up
        (LiH at d=11: H |Q|~2x10⁴ -> E≈+10⁸ Ha).
    on_center_soft
        ``(n_basis_soft,)`` bool mask, same for the softened basis.
    """

    atom_idx: int
    Z: int
    centre_bohr: np.ndarray
    grid: AtomicRadialGrid
    aug_radius: float
    chi_a: np.ndarray
    chi_tilde: Optional[np.ndarray] = None
    on_center_full: Optional[np.ndarray] = None
    on_center_soft: Optional[np.ndarray] = None
    # AO first derivatives on the atomic grid, ``(n_grid, n_basis, 3)`` --
    # needed ONLY for the meta-GGA kinetic-energy-density (t) augmentation
    # (:meth:`GapwAugmentation._compute_atomic_tau`). ``None`` for LDA/GGA
    # runs (not built, to avoid the extra ``evaluate_ao_with_gradient`` cost).
    dchi_a: Optional[np.ndarray] = None
    dchi_tilde: Optional[np.ndarray] = None

    @property
    def n_grid(self) -> int:
        return self.grid.n_points

    @property
    def n_basis(self) -> int:
        return self.chi_a.shape[1]


# ============================================================
# GAPW augmentation orchestrator
# ============================================================


class GapwAugmentation:
    """Per-system GAPW augmentation data and correction builder.

    Owns the per-atom radial grids, the softened basis, and the
    correction build methods for the Hartree and XC contributions.
    Designed to be constructed once per SCF and reused across
    iterations (the atomic grids and AO values are cached).

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet` for the system.
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    grid
        :class:`PlaneWaveGrid` for the smooth FFT Poissonaa.
    lmax
        Maximum multipole order for the r₀ compensator.
    soft_cutoff
        Exponent threshold for the softened basis (bohr⁻^2).
        See :func:`softened_basis`.
    n_radial
        Number of radial grid points per atom.
    lebedev_order
        Lebedev angular grid order.
    smearing_alpha
        Gaussian width for the r₀ multipole compensator (bohr⁻¹).
        ``None`` -> auto-pick from the grid cutoff.
    quiet
        Suppress experimental warnings.
    """

    def __init__(
        self,
        basis,
        system,
        grid: PlaneWaveGrid,
        *,
        lmax: int = _DEFAULT_LMAX,
        soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
        n_radial: int = _DEFAULT_N_RADIAL,
        lebedev_order: int = _DEFAULT_LEBEDEV_ORDER,
        smearing_alpha: Optional[float] = None,
        quiet: bool = False,
    ) -> None:
        # GAPW augmentation correctness bugs fixed (analytic Fock + overlap
        # double-count, 2026-06-26); the experimental-gate warning is retired.
        # `quiet` is kept for API compatibility (no longer gates a warning).
        self._basis = basis
        self._system = system
        self._grid = grid
        self._lmax = lmax
        self._quiet = quiet

        # Build the softened basis for the pseudised density.
        self._soft_basis = softened_basis(basis, system, soft_cutoff=soft_cutoff)

        # If the soft basis is identical to the full basis (e.g. all
        # primitives were pruned and the fallback returned the original),
        # there is no hard/soft distinction -- the augmentation is a no-op.
        # We check this by counting shells: if any shell was dropped or
        # any primitive was actually removed, the number of primitives
        # (sum over shells of len(exponents)) will differ.
        _full_n_prim = sum(len(sh.exponents) for sh in basis.shells())
        _soft_n_prim = sum(len(sh.exponents) for sh in self._soft_basis.shells())
        self._augmentation_active = _soft_n_prim != _full_n_prim

        # Pre-compute the soft->full AO index mapping. When shells are
        # pruned, the soft basis has fewer AOs than the full basis, and
        # the full-basis density matrix must be projected onto the soft
        # subspace before contracting with the soft AO values.
        self._soft_indices = self._compute_soft_to_full_indices(basis)

        # Build the per-atom augmentation data.
        self._atom_data = self._build_atom_data(
            basis,
            system,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
        )

        # Compute the atomic grid volume elements for integration.
        self._voxel_vols = self._build_voxel_volumes()

        # Cached compensator (built on demand from the density).
        self._compensator: Optional[GaussianMultipoleCompensator] = None

    @property
    def n_atoms(self) -> int:
        return len(self._atom_data)

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def atom_data(self) -> list[_AtomAugmentation]:
        """Read-only access to per-atom augmentation data."""
        return list(self._atom_data)

    def _build_voxel_volumes(self) -> list[np.ndarray]:
        """Pre-compute the integration volume for each atomic grid."""
        vols = []
        for ad in self._atom_data:
            # Combined weight w_k . w_l at each (radial, angular) point.
            wt = ad.grid.combined_weights()  # (n_radial, n_angular)
            vols.append(wt)
        return vols

    def _build_atom_data(
        self,
        basis,
        system,
        *,
        n_radial: int,
        lebedev_order: int,
    ) -> list[_AtomAugmentation]:
        """Build the per-atom augmentation data.

        For each atom in the unit cell:
        1. Build the AtomicRadialGrid.
        2. Evaluate the AOs (original and softened) on the grid.
        3. Store in an _AtomAugmentation record.
        """
        atom_list = list(system.unit_cell)
        data_list = []

        # AO -> unit-cell-atom index, for the full and softened bases. Used to
        # restrict the one-centre density n_A¹ to atom-A-centred AOs.
        ao_atom_full = self._ao_atom_index(basis)
        ao_atom_soft = self._ao_atom_index(self._soft_basis)

        for ia, atom in enumerate(atom_list):
            centre = np.asarray(list(atom.xyz), dtype=float)
            Z = int(atom.Z)
            aug_r = estimate_augmentation_radius(Z)
            alpha = default_alpha_for_element(Z)

            # Build the atomic grid. Use a grid that covers the
            # augmentation sphere: extend n_radial to reach aug_r.
            a_grid = AtomicRadialGrid.build(
                centre,
                n_radial=n_radial,
                alpha=alpha,
                lebedev_order=lebedev_order,
            )
            pts = a_grid.cartesian_points().reshape(-1, 3)

            # Evaluate original (hard) AOs.
            chi_a = _core.evaluate_ao(basis, pts)  # (n_grid, n_basis)

            # Evaluate softened AOs (if available).
            try:
                chi_tilde = _core.evaluate_ao(self._soft_basis, pts)
            except Exception:
                chi_tilde = None

            data_list.append(
                _AtomAugmentation(
                    atom_idx=ia,
                    Z=Z,
                    centre_bohr=centre,
                    grid=a_grid,
                    aug_radius=aug_r,
                    chi_a=chi_a,
                    chi_tilde=chi_tilde,
                    on_center_full=(ao_atom_full == ia),
                    on_center_soft=(ao_atom_soft == ia),
                )
            )

        return data_list

    @staticmethod
    def _ao_atom_index(basis) -> np.ndarray:
        """Return an ``(nbasis,)`` int array: the unit-cell atom index that
        each AO is centred on (shell-major order, matching ``evaluate_ao``)."""
        idx: list[int] = []
        for sh in basis.shells():
            idx.extend([int(sh.atom_index)] * GapwAugmentation._n_ao_for_shell(sh))
        return np.array(idx, dtype=np.int64)

    def _build_compensator(
        self,
        density_matrix: np.ndarray,
    ) -> GaussianMultipoleCompensator:
        """Build the GAPW r₀ multipole compensator for D.

        The compensator carries, per atom, the **hard-minus-soft**
        multipoles ``Q_lm = Q_lm[r_a] - Q_lm[r̃_a]`` inside the
        augmentation sphere. Adding r₀ to the soft grid density r̃
        makes ``r̃ + r₀`` reproduce the *true* multipoles of the
        all-electron density that the pruned (soft) basis lost --
        so the smooth FFT-Poisson solve has the correct long-range
        electrostatics, and the per-atom augmentation only has to
        repair the short-range hard/soft difference inside the sphere
        (Lippert, Hutter & Parrinello, *Theor. Chem. Acc.* 103, 124
        (1999), Sec. 2; Krack & Parrinello, *Phys. Chem. Chem. Phys.* 2,
        2105 (2000), Eq. 9 -- the r₀ compensation charge).

        Always rebuilt from the current density matrix so the
        compensator stays consistent with D across SCF iterations.
        Caching would bake in the initial-guess multipoles, which
        drift substantially during the SCF.

        When the augmentation is inactive (soft basis == full basis,
        no pruned core), the hard and soft multipoles are identical
        => Q == 0 => the compensator vanishes, which is correct: there
        is no missing core charge to restore.
        """
        from .periodic_gapw_atomic_grid import compute_multipole_moments

        D = np.asarray(density_matrix, dtype=float)
        alpha = (
            self._smearing_alpha
            if hasattr(self, "_smearing_alpha") and self._smearing_alpha
            else 1.5
        )
        n_atoms = len(self._system.unit_cell)
        n_comp = (self._lmax + 1) ** 2
        Q = np.zeros((n_atoms, n_comp), dtype=float)

        if self._augmentation_active:
            # Hard-minus-soft multipoles of the ONE-CENTRE density, on the
            # same per-atom augmentation grid the SCF uses. _compute_atomic_density
            # is on-centre (atom-A AOs only), so (r_a - r̃_a) is the localized core
            # difference and decays well inside the sphere -- no neighbour leaks in,
            # no r^l blow-up (cf. the old unbounded compute_atom_local_multipoles).
            for ia, ad in enumerate(self._atom_data):
                rho_hard = self._compute_atomic_density(D, ad, use_soft=False)
                rho_soft = self._compute_atomic_density(D, ad, use_soft=True)
                Q[ia] = compute_multipole_moments(
                    rho_hard - rho_soft, ad.grid, self._lmax
                )

        positions = np.array(
            [list(a.xyz) for a in self._system.unit_cell], dtype=float
        )
        comp = GaussianMultipoleCompensator(
            positions=positions,
            alpha=np.full(n_atoms, float(alpha)),
            Q=Q,
            lmax=self._lmax,
        )
        self._compensator = comp
        return comp

    @staticmethod
    def _n_ao_for_shell(sh) -> int:
        """Return the number of AO functions in a shell.

        For spherical (pure) shells the count is 2*l + 1.
        For Cartesian shells it is (l+1)(l+2)/2.
        """
        l = int(sh.l)
        if getattr(sh, "pure", True):
            return 2 * l + 1
        return (l + 1) * (l + 2) // 2

    def _compute_soft_to_full_indices(self, basis) -> np.ndarray:
        """Build the AO index mapping from soft basis -> full basis.

        When shells are pruned from the full basis, the soft basis has
        fewer AOs. This mapping gives ``soft_indices[i] = j`` meaning
        the i-th AO in the soft basis corresponds to the j-th AO in the
        full basis. For non-pruned bases this is the identity.

        Returns
        -------
        indices
            ``(n_basis_soft,)`` int-array of full-basis AO indices.
        """
        if self._soft_basis.nbasis == basis.nbasis:
            return np.arange(basis.nbasis, dtype=np.int64)

        full_shells = basis.shells()
        soft_shells = self._soft_basis.shells()

        # Walk both shell lists in parallel, tracking cumulative AO counts.
        full_ao = 0
        si = 0  # index into soft_shells
        indices: list[int] = []
        for fsh in full_shells:
            n_full = self._n_ao_for_shell(fsh)
            if si < len(soft_shells):
                ssh = soft_shells[si]
                # Match on atom index + angular momentum (same ordering
                # contract: both bases use the same Molecule, so shells
                # appear in the same per-atom, per-l sequence).
                if int(ssh.atom_index) == int(fsh.atom_index) and int(ssh.l) == int(
                    fsh.l
                ):
                    n_soft = self._n_ao_for_shell(ssh)
                    # The soft shell may have the same number of AOs
                    # (pruned primitives, same contraction) or fewer
                    # (shell entirely dropped -> handled by the `else`
                    #  branch that skips this full shell).
                    indices.extend(range(full_ao, full_ao + n_soft))
                    si += 1
                # else: this full shell was pruned -> its AOs are not
                # in the soft basis; skip without adding to indices.
            # else: all remaining full shells are pruned.
            full_ao += n_full
        return np.array(indices, dtype=np.int64)

    def _compute_atomic_density(
        self,
        density_matrix: np.ndarray,
        atom_data: _AtomAugmentation,
        *,
        use_soft: bool = False,
    ) -> np.ndarray:
        """Compute the AO density on a single atom's radial grid.

        Parameters
        ----------
        density_matrix
            ``(n_basis, n_basis)`` AO density matrix (full basis).
        atom_data
            Per-atom augmentation data.
        use_soft
            If True, use the softened AO basis (r̃_a). Otherwise
            use the original (hard) AOs (r_a).

        Returns
        -------
        rho
            ``(n_radial, n_angular)`` density on the atomic grid.
        """
        if use_soft and atom_data.chi_tilde is not None:
            chi = atom_data.chi_tilde
            on_center = atom_data.on_center_soft
        else:
            chi = atom_data.chi_a
            on_center = atom_data.on_center_full
        # Restrict to the ONE-CENTRE density n_A¹: keep only atom-A-centred
        # AOs (Lippert-Hutter-Parrinello 1999, Eq 53). A neighbour's AO,
        # large on A's far grid points, otherwise contaminates n_A¹ and -- via
        # the r^l multipole weight -- blows up the compensator on molecules.
        if on_center is not None:
            chi = chi * on_center[None, :]
        D = np.asarray(density_matrix, dtype=float)
        # When the soft basis has fewer AOs than the full basis
        # (shells pruned), project D onto the soft subspace before
        # contracting with chi_tilde. Without this, the einsum
        # dimension mismatch -> ValueError (GPW-AUDIT-006).
        if use_soft and chi.shape[1] != D.shape[0]:
            idx = self._soft_indices
            D = D[np.ix_(idx, idx)]
        # r(r) = S_muν D_muν . chi_mu(r) . chi_ν(r)
        rho_flat = np.einsum("gm,mn,gn->g", chi, D, chi, optimize=True)
        return rho_flat.reshape(atom_data.grid.n_radial, atom_data.grid.n_angular)

    def _ensure_atomic_gradients(self) -> None:
        """Populate ``dchi_a`` / ``dchi_tilde`` on each atom (lazy).

        AO first derivatives on the atomic grid are needed only for the
        meta-GGA kinetic-energy-density augmentation, so they are built on
        first use (one ``evaluate_ao_with_gradient`` per atom per basis) and
        cached on the per-atom records for the rest of the SCF.
        """
        if getattr(self, "_atomic_gradients_ready", False):
            return
        for ad in self._atom_data:
            pts = ad.grid.cartesian_points().reshape(-1, 3)
            _, gx, gy, gz = _core.evaluate_ao_with_gradient(self._basis, pts)
            ad.dchi_a = np.stack([gx, gy, gz], axis=-1)  # (n_grid, n_basis, 3)
            if ad.chi_tilde is not None:
                _, sx, sy, sz = _core.evaluate_ao_with_gradient(
                    self._soft_basis, pts
                )
                ad.dchi_tilde = np.stack([sx, sy, sz], axis=-1)
        self._atomic_gradients_ready = True

    def _compute_atomic_density_gradient_exact(
        self,
        density_matrix: np.ndarray,
        atom_data: "_AtomAugmentation",
        use_soft: bool,
    ) -> np.ndarray:
        """Exact ``grad rho_A`` on one atom's grid, all three components.

            ``grad rho_A(r) = 2 sum_mn D_mn chi_m(r) grad chi_n(r)``

        The module-level :func:`_atomic_density_gradient` builds only the
        RADIAL part -- its own docstring assumes "a spherically-symmetric
        density r(r)", and it returns ``drho/dr * rhat`` with both angular
        components identically zero. One-centre densities are not
        spherically symmetric in general (an aspherical core is the whole
        reason the l > 0 machinery exists), so for a GGA that under-counts
        ``sigma = |grad rho|^2`` wherever the atomic density has angular
        structure, and the error is largest exactly on the bonded,
        low-symmetry atoms the augmentation is there to fix.

        CP2K forms this analytically for the same reason -- it contracts
        the Clebsch-Gordan derivative coefficients against the analytic
        gradient of the basis-function products rather than differencing
        the density (``qs_vxc_atom.F``).

        Uses the same one-centre masking, soft-basis projection and cached
        ``dchi`` tables as :meth:`_compute_atomic_tau`, so it costs one
        extra contraction and no new integral evaluation.
        """
        self._ensure_atomic_gradients()
        if use_soft and atom_data.chi_tilde is not None:
            chi = atom_data.chi_tilde
            dchi = atom_data.dchi_tilde
            on_center = atom_data.on_center_soft
        else:
            chi = atom_data.chi_a
            dchi = atom_data.dchi_a
            on_center = atom_data.on_center_full
        if on_center is not None:
            chi = chi * on_center[None, :]
            dchi = dchi * on_center[None, :, None]
        D = np.asarray(density_matrix, dtype=float)
        if use_soft and chi.shape[1] != D.shape[0]:
            idx = self._soft_indices
            D = D[np.ix_(idx, idx)]
        chi_D = chi @ D                                    # (n_grid, n_bas)
        n_grid = chi.shape[0]
        grad = np.empty((n_grid, 3), dtype=float)
        for d in range(3):
            grad[:, d] = 2.0 * np.einsum(
                "gm,gm->g", chi_D, dchi[:, :, d], optimize=True
            )
        nr, na = atom_data.grid.n_radial, atom_data.grid.n_angular
        return grad.reshape(nr, na, 3)

    def _compute_atomic_tau(
        self,
        density_matrix: np.ndarray,
        atom_data: _AtomAugmentation,
        use_soft: bool,
    ) -> np.ndarray:
        """Kinetic-energy density t(r) on one atom's radial grid.

            ``t(r) = 1/2 S_muν D_muν S_d dchi_mu/dr_d(r) . dchi_ν/dr_d(r)``

        Same one-centre masking and soft-basis projection convention as
        :meth:`_compute_atomic_density` (the t source is the one-centre
        density's kinetic-energy density). Requires :meth:`_ensure_atomic_gradients`.
        """
        if use_soft and atom_data.dchi_tilde is not None:
            dchi = atom_data.dchi_tilde
            on_center = atom_data.on_center_soft
        else:
            dchi = atom_data.dchi_a
            on_center = atom_data.on_center_full
        if on_center is not None:
            dchi = dchi * on_center[None, :, None]
        D = np.asarray(density_matrix, dtype=float)
        if use_soft and dchi.shape[1] != D.shape[0]:
            idx = self._soft_indices
            D = D[np.ix_(idx, idx)]
        tau_flat = np.zeros(dchi.shape[0], dtype=float)
        for d in range(3):
            gd = dchi[:, :, d]
            tau_flat += np.einsum("gm,mn,gn->g", gd, D, gd, optimize=True)
        tau_flat *= 0.5
        return tau_flat.reshape(atom_data.grid.n_radial, atom_data.grid.n_angular)

    def _project_atomic_vtau(
        self,
        v_tau_hard: np.ndarray,
        v_tau_soft: np.ndarray,
        atom_data: _AtomAugmentation,
        weight: np.ndarray,
    ) -> np.ndarray:
        """Telescoped meta-GGA t Fock block for one atom (full AO basis).

            ``dV_t,muν = 1/2 int v_t^hard dchi_mu.dchi_ν (hard basis)
                       - 1/2 int v_t^soft dchi~_mu.dchi~_ν (soft basis)``

        with ``weight = partition_window * grid_quadrature`` (the same factor
        the scalar r/s channels project with). The soft block is built in the
        soft basis and scattered into the full ``(n_basis, n_basis)`` matrix
        via ``_soft_indices`` (identity when the basis is not pruned). Unlike
        the r/s channels this cannot fold into a single scalar field -- the
        hard and soft t potentials multiply DIFFERENT gradient products
        (dchi vs dchi~), so it is two separate gradient-gradient projections.
        """
        n_basis = self._basis.nbasis
        w = weight.ravel()
        # Hard: full AO gradients.
        wh = (0.5 * v_tau_hard.ravel()) * w
        dchi_h = atom_data.dchi_a  # (n_grid, n_basis, 3)
        M = np.zeros((n_basis, n_basis), dtype=float)
        for d in range(3):
            gd = dchi_h[:, :, d]
            M += (gd * wh[:, None]).T @ gd
        # Soft: soft AO gradients, scattered into the full matrix.
        dchi_s = atom_data.dchi_tilde if atom_data.dchi_tilde is not None else dchi_h
        ws = (0.5 * v_tau_soft.ravel()) * w
        n_soft = dchi_s.shape[1]
        M_soft = np.zeros((n_soft, n_soft), dtype=float)
        for d in range(3):
            gd = dchi_s[:, :, d]
            M_soft += (gd * ws[:, None]).T @ gd
        if n_soft == n_basis:
            M -= M_soft
        else:
            idx = self._soft_indices
            M[np.ix_(idx, idx)] -= M_soft
        return 0.5 * (M + M.T)

    def compute_hartree_correction(
        self,
        density_matrix: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Compute the per-atom Hartree correction to the J matrix.

        For each atom a, compute:

            ``ΔJ_a,muν = ∫_{sphere} chi_mu(r) chi_ν(r) [V_H[r_a](r) - V_H[r̃_a + r₀,a](r)] dr``

        The correction is added to the smooth-grid FFT J.

        Parameters
        ----------
        density_matrix
            ``(n_basis, n_basis)`` AO density matrix.

        Returns
        -------
        corrections
            ``dict[int, np.ndarray]`` mapping atom index to
            ``(n_basis, n_basis)`` ΔJ matrix for that atom.
        """
        if not self._augmentation_active:
            return {}
        D = np.asarray(density_matrix, dtype=float)
        n_basis = D.shape[0]
        corrections: dict[int, np.ndarray] = {}

        for ad in self._atom_data:
            # Compute hard density r_a on the atomic grid.
            rho_a = self._compute_atomic_density(D, ad, use_soft=False)

            # Compute soft density r̃_a on the atomic grid.
            rho_tilde = self._compute_atomic_density(D, ad, use_soft=True)

            # Compute r₀ compensator density on this atomic grid. Own-atom
            # only: the global compensator lives on the smooth grid (the
            # augmentation must not re-add neighbours' compensators).
            rho_0_a = self.compensator_density_on_atomic_grid(
                ad, own_atom_only=True
            )

            # Solve Poisson on the atomic grid for both densities.
            V_hard = solve_poisson_radial(rho_a, ad.grid)
            V_soft = solve_poisson_radial(rho_tilde + rho_0_a, ad.grid)
            delta_V = V_hard - V_soft  # (n_radial, n_angular)

            # Partition-of-unity weight (overlapping spheres do not double-
            # count; reduces to the radial window for an isolated atom).
            window = self._partition_weight(ad)  # (n_radial, n_angular)
            delta_V_windowed = delta_V * window

            # Project back to the AO basis:
            # ΔJ_muν = ∫ chi_mu(r) chi_ν(r) . ΔV(r) . w(r) dr
            # where w(r) is the quadrature weight.
            chi = ad.chi_a  # (n_grid, n_basis)
            wt = ad.grid.combined_weights().ravel()  # (n_grid,)
            delta_V_flat = delta_V_windowed.ravel()  # (n_grid,)

            # ΔJ_muν = S_g chi_mu(g) chi_ν(g) . ΔV(g) . dV(g)
            delta_J = np.einsum(
                "gm,gn,g->mn",
                chi,
                chi,
                delta_V_flat * wt,
                optimize=True,
            )
            delta_J = 0.5 * (delta_J + delta_J.T)
            corrections[ad.atom_idx] = delta_J

        return corrections

    def compute_augmentation_energy(
        self,
        density_matrix: np.ndarray,
        functional=None,
    ) -> tuple[float, float]:
        """Compute the GAPW augmentation energy correction at a given density.

        Returns the total Hartree and XC augmentation energies that must
        be added to the plain GPW total to obtain the all-electron GAPW
        total energy (Krack-Parrinello 2000, doi:10.1039/B001167N).

        The Hartree part is the per-atom difference between the hard-sphere
        and soft-sphere Hartree energies, built from the same hard/soft
        densities and Poisson solves that drive the augmented J matrix.

        The XC part is the per-atom difference ``E_xc[r_a] - E_xc[r̃_a]``
        evaluated on each atomic radial grid.

        Parameters
        ----------
        density_matrix
            ``(n_basis, n_basis)`` AO density matrix.
        functional
            :class:`vibeqc._vibeqc_core.Functional` instance, or None
            for Hartree-Fock (XC part returns 0).

        Returns
        -------
        (e_hartree_aug, e_xc_aug)
            Hartree and XC augmentation energies in Hartree.
        """
        if not self._augmentation_active:
            return 0.0, 0.0
        D = np.asarray(density_matrix, dtype=float)
        e_hartree_aug = 0.0
        e_xc_aug = 0.0

        for ad in self._atom_data:
            # Hard density on the atomic grid.
            rho_a = self._compute_atomic_density(D, ad, use_soft=False)
            # Soft density on the atomic grid.
            rho_tilde = self._compute_atomic_density(D, ad, use_soft=True)
            # Compensator density (own-atom only; see build_J / the
            # compensator_density_on_atomic_grid docstring).
            rho_0_a = self.compensator_density_on_atomic_grid(
                ad, own_atom_only=True
            )

            # Hartree energy augmentation:
            #   E_H_aug = 1/2 S_a (∫ r_a V_H[r_a] dr - ∫ (r̃_a + r₀,a) V_H[r̃_a + r₀,a] dr)
            # NOT (a-b)(c-d) which double-counts the cross terms.
            V_hard = solve_poisson_radial(rho_a, ad.grid)
            V_soft = solve_poisson_radial(rho_tilde + rho_0_a, ad.grid)
            wt = ad.grid.combined_weights()
            # Partition-of-unity weight (overlapping spheres do not double-
            # count; reduces to the radial window for an isolated atom). Same
            # weight the analytic Fock uses, so 1/2 tr(D.J) = E_H holds.
            window = self._partition_weight(ad)
            e_hard = 0.5 * float(np.einsum("ra,ra,ra->", rho_a, V_hard * window, wt))
            e_soft = 0.5 * float(
                np.einsum("ra,ra,ra->", (rho_tilde + rho_0_a), V_soft * window, wt)
            )
            e_hartree_aug += e_hard - e_soft

            # XC energy augmentation: E_xc[r_a] - E_xc[r̃_a].
            if functional is not None:
                rho_hard_flat = rho_a.ravel()
                rho_soft_flat = rho_tilde.ravel()
                # Evaluate per-grid-point XC energy density via
                # the libxc functional's eval_unpolarised API.
                is_mgga = functional.kind == _core.XCKind.MGGA
                is_gga = functional.kind != _core.XCKind.LDA and not is_mgga
                if is_gga or is_mgga:
                    grad_a = self._compute_atomic_density_gradient_exact(
                        D, ad, use_soft=False)
                    grad_tilde = self._compute_atomic_density_gradient_exact(
                        D, ad, use_soft=True)
                    sigma_a = (grad_a**2).sum(axis=-1)
                    sigma_tilde = (grad_tilde**2).sum(axis=-1)
                if is_mgga:
                    self._ensure_atomic_gradients()
                    tau_a = self._compute_atomic_tau(D, ad, use_soft=False)
                    tau_t = self._compute_atomic_tau(D, ad, use_soft=True)
                    tau_a = _von_weizsaecker_floor(tau_a, rho_a, sigma_a)
                    tau_t = _von_weizsaecker_floor(tau_t, rho_tilde, sigma_tilde)
                    exc_hard, *_ = functional.eval_unpolarised_mgga(
                        rho_hard_flat, sigma_a.ravel(), tau_a.ravel()
                    )
                    exc_soft, *_ = functional.eval_unpolarised_mgga(
                        rho_soft_flat, sigma_tilde.ravel(), tau_t.ravel()
                    )
                elif is_gga:
                    exc_hard, _, _ = functional.eval_unpolarised(
                        rho_hard_flat, sigma_a.ravel()
                    )
                    exc_soft, _, _ = functional.eval_unpolarised(
                        rho_soft_flat, sigma_tilde.ravel()
                    )
                else:
                    # LDA: pass a dummy sigma array (libxc requires it).
                    dummy_sigma = np.zeros_like(rho_hard_flat)
                    exc_hard, _, _ = functional.eval_unpolarised(
                        rho_hard_flat, dummy_sigma
                    )
                    exc_soft, _, _ = functional.eval_unpolarised(
                        rho_soft_flat, dummy_sigma
                    )
                exc_hard = np.asarray(exc_hard).reshape(rho_a.shape)
                exc_soft = np.asarray(exc_soft).reshape(rho_tilde.shape)
                # e_xc per unit volume; integrate with weights.
                e_xc_aug += float(
                    np.einsum("ra,ra->", (exc_hard - exc_soft) * window, wt)
                )

        return e_hartree_aug, e_xc_aug

    def compute_augmentation_energy_polarised(
        self,
        D_alpha: np.ndarray,
        D_beta: np.ndarray,
        functional,
    ) -> tuple[float, float]:
        """Spin-polarised GAPW augmentation energy correction.

        The Hartree augmentation depends only on the total density and is
        identical to the restricted path. The XC part uses libxc's spin=2
        interface on the hard and soft per-spin atomic-grid densities:
        ``E_xc[r_a↑, r_a↓] - E_xc[r̃_a↑, r̃_a↓]``.
        """
        if not self._augmentation_active:
            return 0.0, 0.0

        D_a = np.asarray(D_alpha, dtype=float)
        D_b = np.asarray(D_beta, dtype=float)
        e_hartree_aug, _ = self.compute_augmentation_energy(
            D_a + D_b,
            functional=None,
        )
        e_xc_aug = 0.0
        is_mgga = functional.kind == _core.XCKind.MGGA
        is_gga = functional.kind != _core.XCKind.LDA and not is_mgga
        if is_mgga:
            self._ensure_atomic_gradients()

        for ad in self._atom_data:
            rho_a_alpha = self._compute_atomic_density(D_a, ad, use_soft=False)
            rho_a_beta = self._compute_atomic_density(D_b, ad, use_soft=False)
            rho_t_alpha = self._compute_atomic_density(D_a, ad, use_soft=True)
            rho_t_beta = self._compute_atomic_density(D_b, ad, use_soft=True)

            if is_gga or is_mgga:
                grad_a_a = self._compute_atomic_density_gradient_exact(
                    D_a, ad, use_soft=False)
                grad_a_b = self._compute_atomic_density_gradient_exact(
                    D_b, ad, use_soft=False)
                grad_t_a = self._compute_atomic_density_gradient_exact(
                    D_a, ad, use_soft=True)
                grad_t_b = self._compute_atomic_density_gradient_exact(
                    D_b, ad, use_soft=True)

                sigma_aa_a = (grad_a_a**2).sum(axis=-1)
                sigma_ab_a = (grad_a_a * grad_a_b).sum(axis=-1)
                sigma_bb_a = (grad_a_b**2).sum(axis=-1)
                sigma_aa_t = (grad_t_a**2).sum(axis=-1)
                sigma_ab_t = (grad_t_a * grad_t_b).sum(axis=-1)
                sigma_bb_t = (grad_t_b**2).sum(axis=-1)
            else:
                sigma_aa_a = sigma_ab_a = sigma_bb_a = np.zeros_like(rho_a_alpha)
                sigma_aa_t = sigma_ab_t = sigma_bb_t = np.zeros_like(rho_t_alpha)

            if is_mgga:
                tau_a_a = self._compute_atomic_tau(D_a, ad, use_soft=False)
                tau_a_b = self._compute_atomic_tau(D_b, ad, use_soft=False)
                tau_t_a = self._compute_atomic_tau(D_a, ad, use_soft=True)
                tau_t_b = self._compute_atomic_tau(D_b, ad, use_soft=True)
                tau_a_a = _von_weizsaecker_floor(tau_a_a, rho_a_alpha, sigma_aa_a)
                tau_a_b = _von_weizsaecker_floor(tau_a_b, rho_a_beta, sigma_bb_a)
                tau_t_a = _von_weizsaecker_floor(tau_t_a, rho_t_alpha, sigma_aa_t)
                tau_t_b = _von_weizsaecker_floor(tau_t_b, rho_t_beta, sigma_bb_t)
                exc_hard, *_ = functional.eval_polarised_mgga(
                    np.maximum(rho_a_alpha.ravel(), 0.0),
                    np.maximum(rho_a_beta.ravel(), 0.0),
                    sigma_aa_a.ravel(),
                    sigma_ab_a.ravel(),
                    sigma_bb_a.ravel(),
                    tau_a_a.ravel(),
                    tau_a_b.ravel(),
                )
                exc_soft, *_ = functional.eval_polarised_mgga(
                    np.maximum(rho_t_alpha.ravel(), 0.0),
                    np.maximum(rho_t_beta.ravel(), 0.0),
                    sigma_aa_t.ravel(),
                    sigma_ab_t.ravel(),
                    sigma_bb_t.ravel(),
                    tau_t_a.ravel(),
                    tau_t_b.ravel(),
                )
            else:
                exc_hard, *_ = functional.eval_polarised(
                    np.maximum(rho_a_alpha.ravel(), 0.0),
                    np.maximum(rho_a_beta.ravel(), 0.0),
                    sigma_aa_a.ravel(),
                    sigma_ab_a.ravel(),
                    sigma_bb_a.ravel(),
                )
                exc_soft, *_ = functional.eval_polarised(
                    np.maximum(rho_t_alpha.ravel(), 0.0),
                    np.maximum(rho_t_beta.ravel(), 0.0),
                    sigma_aa_t.ravel(),
                    sigma_ab_t.ravel(),
                    sigma_bb_t.ravel(),
                )
            exc_hard = np.asarray(exc_hard).reshape(rho_a_alpha.shape)
            exc_soft = np.asarray(exc_soft).reshape(rho_t_alpha.shape)
            window = self._partition_weight(ad)
            wt = ad.grid.combined_weights()
            e_xc_aug += float(
                np.einsum("ra,ra->", (exc_hard - exc_soft) * window, wt)
            )

        return e_hartree_aug, e_xc_aug

    def compute_xc_correction(
        self,
        density_matrix: np.ndarray,
        functional,
        *,
        return_energy: bool = False,
    ) -> dict[int, np.ndarray] | tuple[dict[int, np.ndarray], float]:
        """Compute the per-atom XC correction to the V_xc matrix.

        For each atom a:

            ``ΔV_xc,a,muν = ∫ chi_mu chi_ν [v_xc[r_a] - v_xc[r̃_a]] dr``

        Parameters
        ----------
        density_matrix
            ``(n_basis, n_basis)`` AO density matrix.
        functional
            :class:`vibeqc._vibeqc_core.Functional` instance.

        Returns
        -------
        corrections
            ``dict[int, np.ndarray]`` mapping atom index to
            ``(n_basis, n_basis)`` ΔV_xc matrix.
            With ``return_energy=True``, returns
            ``(corrections, e_xc_augmentation)`` so an SCF iteration can use
            the energy density already evaluated for the Fock instead of
            repeating the atomic-grid XC work.
        """
        if not self._augmentation_active:
            return ({}, 0.0) if return_energy else {}
        D = np.asarray(density_matrix, dtype=float)
        n_basis = D.shape[0]
        corrections: dict[int, np.ndarray] = {}
        e_xc_aug = 0.0

        is_mgga = functional.kind == _core.XCKind.MGGA
        is_gga = functional.kind != _core.XCKind.LDA and not is_mgga
        if is_mgga:
            self._ensure_atomic_gradients()

        for ad in self._atom_data:
            # Hard density
            rho_a = self._compute_atomic_density(D, ad, use_soft=False)
            # Soft density
            rho_tilde = self._compute_atomic_density(D, ad, use_soft=True)

            if is_gga or is_mgga:
                # Gradient for the sigma channel (finite differences on the
                # atomic radial grid).
                grad_a = self._compute_atomic_density_gradient_exact(
                    D, ad, use_soft=False)
                grad_tilde = self._compute_atomic_density_gradient_exact(
                    D, ad, use_soft=True)
                sigma_a = (grad_a**2).sum(axis=-1)
                sigma_tilde = (grad_tilde**2).sum(axis=-1)
            else:
                sigma_a = np.zeros_like(rho_a)
                sigma_tilde = np.zeros_like(rho_tilde)

            # Evaluate XC potential on the atomic grid.
            rho_flat_a = np.maximum(rho_a.ravel(), 0.0)
            rho_flat_tilde = np.maximum(rho_tilde.ravel(), 0.0)

            v_tau_hard = v_tau_soft = None
            if is_mgga:
                tau_a = self._compute_atomic_tau(D, ad, use_soft=False)
                tau_t = self._compute_atomic_tau(D, ad, use_soft=True)
                tau_a = _von_weizsaecker_floor(tau_a, rho_a, sigma_a)
                tau_t = _von_weizsaecker_floor(tau_t, rho_tilde, sigma_tilde)
                (
                    exc_a,
                    v_rho_a,
                    v_sigma_a,
                    v_tau_a,
                ) = functional.eval_unpolarised_mgga(
                    rho_flat_a,
                    sigma_a.ravel(),
                    tau_a.ravel(),
                )
                (
                    exc_t,
                    v_rho_t,
                    v_sigma_t,
                    v_tau_t,
                ) = functional.eval_unpolarised_mgga(
                    rho_flat_tilde,
                    sigma_tilde.ravel(),
                    tau_t.ravel(),
                )
                v_tau_hard = np.asarray(v_tau_a).reshape(rho_a.shape)
                v_tau_soft = np.asarray(v_tau_t).reshape(rho_tilde.shape)
            else:
                exc_a, v_rho_a, v_sigma_a = functional.eval_unpolarised(
                    rho_flat_a,
                    sigma_a.ravel(),
                )
                exc_t, v_rho_t, v_sigma_t = functional.eval_unpolarised(
                    rho_flat_tilde,
                    sigma_tilde.ravel(),
                )

            v_xc_hard = np.asarray(v_rho_a).reshape(rho_a.shape)
            v_xc_soft = np.asarray(v_rho_t).reshape(rho_tilde.shape)
            delta_v_xc = v_xc_hard - v_xc_soft

            # For GGA / meta-GGA: include the sigma-derivative contribution.
            if is_gga or is_mgga:
                v_s_hard = np.asarray(v_sigma_a).reshape(rho_a.shape)
                v_s_soft = np.asarray(v_sigma_t).reshape(rho_tilde.shape)
                # ΔV_xc = ∫ chi_mu chi_ν . Δv_rho + 2 Δv_sigma . gradr . grad(chi_mu chi_ν)
                # We approximate this on the radial grid by adding
                # the divergence term.
                # v_s_* is the scalar v_sigma (n_radial, n_ang); grad_* carries
                # a trailing Cartesian axis (n_radial, n_ang, 3). Broadcast the
                # scalar onto the vector -- mirrors the GPW path's
                # ``2.0 * v_sigma_grid[..., None] * grad_rho`` in
                # periodic_gapw_j.py:_project_vxc_to_ao.
                flux_hard = 2.0 * v_s_hard[..., None] * grad_a
                flux_soft = 2.0 * v_s_soft[..., None] * grad_tilde
                div_hard = _atomic_divergence(flux_hard, ad.grid)
                div_soft = _atomic_divergence(flux_soft, ad.grid)
                delta_v_xc -= div_hard - div_soft

            # Partition-of-unity weight + project (same weight as the Hartree
            # augmentation / the analytic Fock; no double-count for overlapping
            # spheres, reduces to the radial window for an isolated atom).
            window = self._partition_weight(ad)
            delta_v_windowed = delta_v_xc * window

            chi = ad.chi_a
            weights = ad.grid.combined_weights()
            e_xc_aug += float(
                np.einsum(
                    "ra,ra->",
                    (
                        np.asarray(exc_a).reshape(rho_a.shape)
                        - np.asarray(exc_t).reshape(rho_tilde.shape)
                    )
                    * window,
                    weights,
                )
            )
            wt = weights.ravel()
            dv_flat = delta_v_windowed.ravel()

            delta_Vxc = np.einsum(
                "gm,gn,g->mn",
                chi,
                chi,
                dv_flat * wt,
                optimize=True,
            )
            delta_Vxc = 0.5 * (delta_Vxc + delta_Vxc.T)

            # Meta-GGA t channel: 1/2 int Dv_tau dchi.dchi (telescoped, a
            # gradient-gradient projection that cannot fold into the scalar
            # delta_v field above -- see _project_atomic_vtau).
            if is_mgga:
                delta_Vxc = delta_Vxc + self._project_atomic_vtau(
                    v_tau_hard, v_tau_soft, ad, window * ad.grid.combined_weights()
                )

            corrections[ad.atom_idx] = delta_Vxc

        if return_energy:
            return corrections, e_xc_aug
        return corrections

    def compute_xc_correction_polarised(
        self,
        D_alpha: np.ndarray,
        D_beta: np.ndarray,
        functional,
        *,
        return_energy: bool = False,
    ) -> (
        tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
        | tuple[dict[int, np.ndarray], dict[int, np.ndarray], float]
    ):
        """Per-atom XC correction for spin-polarised (UHF/UKS) densities.

        For each atom a, computes the correction to the V_xc matrix for
        both spin channels:

            ``ΔV_xc,muν↑ = ∫ chi_mu chi_ν [v_xc↑[r_a↑, r_a↓] - v_xc↑[r̃_a↑, r̃_a↓]] dr``

            ``ΔV_xc,muν↓ = ∫ chi_mu chi_ν [v_xc↓[r_a↑, r_a↓] - v_xc↓[r̃_a↑, r̃_a↓]] dr``

        The smooth-grid V_xc is then augmented by adding these per-atom
        corrections to each spin channel separately.

        Parameters
        ----------
        D_alpha, D_beta
            ``(n_basis, n_basis)`` per-spin AO density matrices.
        functional
            :class:`vibeqc._vibeqc_core.Functional` instance (spin=2
            polarised path).

        Returns
        -------
        corrections_alpha, corrections_beta
            Two ``dict[int, np.ndarray]`` mapping atom index to
            ``(n_basis, n_basis)`` ΔV_xc matrix for a / b spin.
            With ``return_energy=True``, the spin-polarised atomic XC
            augmentation energy is returned as a third value.
        """
        if not self._augmentation_active:
            return ({}, {}, 0.0) if return_energy else ({}, {})
        D_a = np.asarray(D_alpha, dtype=float)
        D_b = np.asarray(D_beta, dtype=float)
        n_basis = D_a.shape[0]
        corrections_alpha: dict[int, np.ndarray] = {}
        corrections_beta: dict[int, np.ndarray] = {}
        e_xc_aug = 0.0

        is_mgga = functional.kind == _core.XCKind.MGGA
        is_gga = functional.kind != _core.XCKind.LDA and not is_mgga
        if is_mgga:
            self._ensure_atomic_gradients()

        for ad in self._atom_data:
            # Hard atomic densities per spin.
            rho_a_alpha = self._compute_atomic_density(D_a, ad, use_soft=False)
            rho_a_beta = self._compute_atomic_density(D_b, ad, use_soft=False)
            # Soft atomic densities per spin.
            rho_t_alpha = self._compute_atomic_density(D_a, ad, use_soft=True)
            rho_t_beta = self._compute_atomic_density(D_b, ad, use_soft=True)

            if is_gga or is_mgga:
                grad_a_a = self._compute_atomic_density_gradient_exact(
                    D_a, ad, use_soft=False)
                grad_a_b = self._compute_atomic_density_gradient_exact(
                    D_b, ad, use_soft=False)
                grad_t_a = self._compute_atomic_density_gradient_exact(
                    D_a, ad, use_soft=True)
                grad_t_b = self._compute_atomic_density_gradient_exact(
                    D_b, ad, use_soft=True)

                sigma_aa_a = (grad_a_a**2).sum(axis=-1)
                sigma_ab_a = (grad_a_a * grad_a_b).sum(axis=-1)
                sigma_bb_a = (grad_a_b**2).sum(axis=-1)
                sigma_aa_t = (grad_t_a**2).sum(axis=-1)
                sigma_ab_t = (grad_t_a * grad_t_b).sum(axis=-1)
                sigma_bb_t = (grad_t_b**2).sum(axis=-1)
            else:
                sigma_aa_a = sigma_ab_a = sigma_bb_a = np.zeros_like(rho_a_alpha)
                sigma_aa_t = sigma_ab_t = sigma_bb_t = np.zeros_like(rho_t_alpha)

            # Evaluate polarised XC on the atomic grid.
            ra_f = np.maximum(rho_a_alpha.ravel(), 0.0)
            rb_f = np.maximum(rho_a_beta.ravel(), 0.0)
            rt_a_f = np.maximum(rho_t_alpha.ravel(), 0.0)
            rt_b_f = np.maximum(rho_t_beta.ravel(), 0.0)

            v_tau_h_a = v_tau_h_b = v_tau_s_a = v_tau_s_b = None
            if is_mgga:
                tau_a_a = self._compute_atomic_tau(D_a, ad, use_soft=False)
                tau_a_b = self._compute_atomic_tau(D_b, ad, use_soft=False)
                tau_t_a = self._compute_atomic_tau(D_a, ad, use_soft=True)
                tau_t_b = self._compute_atomic_tau(D_b, ad, use_soft=True)
                tau_a_a = _von_weizsaecker_floor(tau_a_a, rho_a_alpha, sigma_aa_a)
                tau_a_b = _von_weizsaecker_floor(tau_a_b, rho_a_beta, sigma_bb_a)
                tau_t_a = _von_weizsaecker_floor(tau_t_a, rho_t_alpha, sigma_aa_t)
                tau_t_b = _von_weizsaecker_floor(tau_t_b, rho_t_beta, sigma_bb_t)
                (
                    exc_a,
                    v_rho_a_a,
                    v_rho_a_b,
                    v_s_aa_a,
                    v_s_ab_a,
                    v_s_bb_a,
                    v_tau_a_a,
                    v_tau_a_b,
                ) = functional.eval_polarised_mgga(
                    ra_f,
                    rb_f,
                    sigma_aa_a.ravel(),
                    sigma_ab_a.ravel(),
                    sigma_bb_a.ravel(),
                    tau_a_a.ravel(),
                    tau_a_b.ravel(),
                )
                (
                    exc_t,
                    v_rho_t_a,
                    v_rho_t_b,
                    v_s_aa_t,
                    v_s_ab_t,
                    v_s_bb_t,
                    v_tau_t_a,
                    v_tau_t_b,
                ) = functional.eval_polarised_mgga(
                    rt_a_f,
                    rt_b_f,
                    sigma_aa_t.ravel(),
                    sigma_ab_t.ravel(),
                    sigma_bb_t.ravel(),
                    tau_t_a.ravel(),
                    tau_t_b.ravel(),
                )
                v_tau_h_a = np.asarray(v_tau_a_a).reshape(rho_a_alpha.shape)
                v_tau_h_b = np.asarray(v_tau_a_b).reshape(rho_a_beta.shape)
                v_tau_s_a = np.asarray(v_tau_t_a).reshape(rho_t_alpha.shape)
                v_tau_s_b = np.asarray(v_tau_t_b).reshape(rho_t_beta.shape)
            else:
                (exc_a, v_rho_a_a, v_rho_a_b, v_s_aa_a, v_s_ab_a, v_s_bb_a) = (
                    functional.eval_polarised(
                        ra_f,
                        rb_f,
                        sigma_aa_a.ravel(),
                        sigma_ab_a.ravel(),
                        sigma_bb_a.ravel(),
                    )
                )
                (exc_t, v_rho_t_a, v_rho_t_b, v_s_aa_t, v_s_ab_t, v_s_bb_t) = (
                    functional.eval_polarised(
                        rt_a_f,
                        rt_b_f,
                        sigma_aa_t.ravel(),
                        sigma_ab_t.ravel(),
                        sigma_bb_t.ravel(),
                    )
                )

            # Un-flatten potentials.
            v_hard_a = np.asarray(v_rho_a_a).reshape(rho_a_alpha.shape)
            v_hard_b = np.asarray(v_rho_a_b).reshape(rho_a_beta.shape)
            v_soft_a = np.asarray(v_rho_t_a).reshape(rho_t_alpha.shape)
            v_soft_b = np.asarray(v_rho_t_b).reshape(rho_t_beta.shape)

            delta_v_a = v_hard_a - v_soft_a  # a channel
            delta_v_b = v_hard_b - v_soft_b  # b channel

            # GGA / meta-GGA: include s-derivative (divergence) contributions.
            if is_gga or is_mgga:
                vs_aa_h = np.asarray(v_s_aa_a).reshape(rho_a_alpha.shape)
                vs_ab_h = v_s_ab_a.reshape(rho_a_alpha.shape)
                vs_bb_h = v_s_bb_a.reshape(rho_a_alpha.shape)
                vs_aa_s = v_s_aa_t.reshape(rho_t_alpha.shape)
                vs_ab_s = v_s_ab_t.reshape(rho_t_alpha.shape)
                vs_bb_s = v_s_bb_t.reshape(rho_t_alpha.shape)

                # a channel: flux = 2(v_saa . gradr_a + v_sab . gradr_b)
                # vs_* are scalars (n_radial, n_ang); grad_* carry a trailing
                # Cartesian axis (..., 3) -- broadcast the scalar onto the vector
                # (mirrors periodic_gapw_j.py:_project_vxc_to_ao).
                flux_hard_a = 2.0 * (
                    vs_aa_h[..., None] * grad_a_a + vs_ab_h[..., None] * grad_a_b
                )
                flux_soft_a = 2.0 * (
                    vs_aa_s[..., None] * grad_t_a + vs_ab_s[..., None] * grad_t_b
                )
                div_hard_a = _atomic_divergence(flux_hard_a, ad.grid)
                div_soft_a = _atomic_divergence(flux_soft_a, ad.grid)
                delta_v_a -= div_hard_a - div_soft_a

                # b channel: flux = 2(v_sab . gradr_a + v_sbb . gradr_b)
                flux_hard_b = 2.0 * (
                    vs_ab_h[..., None] * grad_a_a + vs_bb_h[..., None] * grad_a_b
                )
                flux_soft_b = 2.0 * (
                    vs_ab_s[..., None] * grad_t_a + vs_bb_s[..., None] * grad_t_b
                )
                div_hard_b = _atomic_divergence(flux_hard_b, ad.grid)
                div_soft_b = _atomic_divergence(flux_soft_b, ad.grid)
                delta_v_b -= div_hard_b - div_soft_b

            # Partition-of-unity weight + project (overlapping spheres do not
            # double-count; reduces to the radial window for an isolated atom).
            window = self._partition_weight(ad)
            dv_a_w = delta_v_a * window
            dv_b_w = delta_v_b * window

            chi = ad.chi_a
            weights = ad.grid.combined_weights()
            e_xc_aug += float(
                np.einsum(
                    "ra,ra->",
                    (
                        np.asarray(exc_a).reshape(rho_a_alpha.shape)
                        - np.asarray(exc_t).reshape(rho_t_alpha.shape)
                    )
                    * window,
                    weights,
                )
            )
            wt = weights.ravel()

            delta_Va = np.einsum(
                "gm,gn,g->mn",
                chi,
                chi,
                dv_a_w.ravel() * wt,
                optimize=True,
            )
            delta_Va = 0.5 * (delta_Va + delta_Va.T)

            delta_Vb = np.einsum(
                "gm,gn,g->mn",
                chi,
                chi,
                dv_b_w.ravel() * wt,
                optimize=True,
            )
            delta_Vb = 0.5 * (delta_Vb + delta_Vb.T)

            # Meta-GGA per-spin t channel: 1/2 int Dv_tau_s dchi.dchi
            # (telescoped gradient-gradient projection; see _project_atomic_vtau).
            if is_mgga:
                pw = window * ad.grid.combined_weights()
                delta_Va = delta_Va + self._project_atomic_vtau(
                    v_tau_h_a, v_tau_s_a, ad, pw
                )
                delta_Vb = delta_Vb + self._project_atomic_vtau(
                    v_tau_h_b, v_tau_s_b, ad, pw
                )

            corrections_alpha[ad.atom_idx] = delta_Va
            corrections_beta[ad.atom_idx] = delta_Vb

        if return_energy:
            return corrections_alpha, corrections_beta, e_xc_aug
        return corrections_alpha, corrections_beta

    def compensator_density_on_atomic_grid(
        self,
        atom_data: _AtomAugmentation,
        *,
        own_atom_only: bool = False,
    ) -> np.ndarray:
        """Evaluate the r₀ compensator density on an atomic grid.

        Parameters
        ----------
        own_atom_only
            If True, include only the compensator centred on *this* atom
            (``atom_data.atom_idx``), not the global sum over all atoms.

            This is the **physically correct** choice for the per-atom
            augmentation (Krack & Parrinello 2000): the augmentation
            ``E_H^a[ñ_a + n_{0,a}]`` must use atom a's OWN compensator
            ``n_{0,a}`` only. The *global* compensator ``n_0 = Σ_b n_{0,b}``
            belongs on the smooth FFT grid (``V_smooth``); summing all atoms'
            Gaussians on each atom's radial grid and feeding them to the
            isolated spherical ``solve_poisson_radial`` double-counts the
            inter-atomic compensator interaction that the smooth term already
            holds -- the dominant part of the bonded-molecule augmentation
            over-bind (H₂/STO-3G ~+66 mHa at fixed density). For an isolated
            atom the global and own-atom compensators coincide, so single-atom
            results (He/Ne/O) are unchanged.

        Returns
        -------
        rho_0
            ``(n_radial, n_angular)`` compensator density on the
            atomic grid. Zero if no compensator has been built.
        """
        if self._compensator is None:
            return np.zeros((atom_data.grid.n_radial, atom_data.grid.n_angular))

        # Evaluate the compensator at the atomic grid points.
        pts = atom_data.grid.cartesian_points().reshape(-1, 3)
        rho_0_flat = np.zeros(pts.shape[0], dtype=float)

        L = self._grid.lattice_bohr
        for ia in range(self._compensator.n_atoms):
            if own_atom_only and ia != atom_data.atom_idx:
                continue
            R_a = self._compensator.positions[ia]
            alpha = float(self._compensator.alpha[ia])
            Q_a = self._compensator.Q[ia]

            delta = pts - R_a
            r2 = (delta * delta).sum(axis=-1)
            r = np.sqrt(np.maximum(r2, 1e-30))
            gauss = np.exp(-(alpha**2) * r2)
            r_safe = np.where(r > 1e-30, r, 1.0)
            omega = delta / r_safe[:, None]

            from .periodic_gapw_atomic_grid import real_spherical_harmonics

            S = real_spherical_harmonics(
                omega, self._compensator.lmax
            )  # (n_comp, n_pts)

            for l in range(self._compensator.lmax + 1):
                from .periodic_gapw_smearing import _multipole_normalisation as N_lm

                Nl = N_lm(l, alpha)
                r_l = r**l
                for m in range(-l, l + 1):
                    idx = multipole_index(l, m)
                    rho_0_flat += Q_a[idx] * Nl * r_l * S[idx] * gauss

        return rho_0_flat.reshape(atom_data.grid.n_radial, atom_data.grid.n_angular)

    def _partition_weight(self, atom_data: _AtomAugmentation) -> np.ndarray:
        """Partition-of-unity weight on atom A's grid for overlapping spheres.

        The per-atom augmentation integrates the hard/soft difference over
        atom A's sphere with a radial window ``w_A(r)`` that tapers 1 → 0 at
        the augmentation radius. When two atoms' spheres overlap (bonded
        atoms), the physical overlap region is corrected by BOTH spheres, so
        the augmentation over-binds (H₂/STO-3G: the over-bind scales with the
        sphere overlap ``2·r_aug/d`` — −44 mHa at d=1.4, → 0 once the spheres
        separate). The compensator-on-own-atom fix removes the spurious
        per-atom compensator interaction; this partition removes the remaining
        spatial double-count.

        The weight distributes the *union* window ``W(r) = max_b w_b`` among
        the overlapping atoms in proportion to their individual windows:

            ``p_A(r) = w_A(r) · max_b w_b(r) / Σ_b w_b(r)``

        so ``Σ_A p_A`` over the overlapping atoms equals the union window ``W``
        (no region is counted more than its union taper). For an **isolated**
        atom ``Σ_b w_b = max_b w_b = w_A`` ⇒ ``p_A = w_A`` exactly — single-atom
        results (He/Ne/O) are unchanged. Purely geometric (no D dependence),
        so it is cached per atom.

        Returns
        -------
        p
            ``(n_radial, n_angular)`` partition weight on atom A's grid.
        """
        if not hasattr(self, "_partition_cache"):
            self._partition_cache: dict[int, np.ndarray] = {}
        ai = atom_data.atom_idx
        cached = self._partition_cache.get(ai)
        if cached is not None:
            return cached
        nr, na = atom_data.grid.n_radial, atom_data.grid.n_angular
        pts = atom_data.grid.cartesian_points().reshape(nr, na, 3)
        w_sum = np.zeros((nr, na))
        w_max = np.zeros((nr, na))
        w_self = np.zeros((nr, na))
        for adb in self._atom_data:
            d_b = np.linalg.norm(
                pts - np.asarray(adb.centre_bohr)[None, None, :], axis=-1
            )
            w_b = _radial_window(d_b.ravel(), adb.aug_radius, width=0.5).reshape(
                nr, na
            )
            w_sum += w_b
            w_max = np.maximum(w_max, w_b)
            if adb.atom_idx == ai:
                w_self = w_b
        denom = np.where(w_sum > 1e-12, w_sum, 1.0)
        part = np.where(w_sum > 1e-12, w_self * w_max / denom, 0.0)
        self._partition_cache[ai] = part
        return part


# ============================================================
# Radial helper functions
# ============================================================


def _radial_window(
    r: np.ndarray,
    cutoff: float,
    width: float = 0.5,
) -> np.ndarray:
    """Smooth windowing function that goes from 1 -> 0 near ``cutoff``.

    Uses a cosine taper: f(r) = 1 for r < cutoff - width,
    f(r) = 0 for r > cutoff, and a smooth cosine transition
    in between.

    Parameters
    ----------
    r
        Radial coordinates (bohr).
    cutoff
        Cutoff radius (bohr).
    width
        Transition width (bohr).

    Returns
    -------
    window
        Array of shape ``r.shape`` with values in [0, 1].
    """
    window = np.ones_like(r)
    mask = r > (cutoff - width)
    transition = (cutoff - r[mask]) / width
    window[mask] = np.clip(transition, 0.0, 1.0)
    # Apply a smoother cosine taper for the transition region.
    taper = r > (cutoff - width)
    t = (r[taper] - (cutoff - width)) / width
    window[taper] = 0.5 * (1.0 + np.cos(math.pi * np.clip(t, 0.0, 1.0)))
    window[r > cutoff] = 0.0
    return window


def _atomic_density_gradient(
    rho: np.ndarray,
    grid: AtomicRadialGrid,
) -> np.ndarray:
    """Compute the gradient of a density on an atomic grid.

    Uses finite differences in the radial direction. For a
    spherically-symmetric density r(r) on the radial x angular grid,
    the gradient is dr/dr . r̂.

    Parameters
    ----------
    rho
        ``(n_radial, n_angular)`` density.
    grid
        Atomic radial grid.

    Returns
    -------
    grad
        ``(n_radial, n_angular, 3)`` gradient vectors.
    """
    nr, na = rho.shape
    r = grid.r
    omega = grid.angular_xyz  # (n_angular, 3)

    # Radial derivative via finite differences.
    drho_dr = np.zeros_like(rho)
    for ik in range(1, nr - 1):
        dr = r[ik + 1] - r[ik - 1]
        if dr > 1e-14:
            drho_dr[ik] = (rho[ik + 1] - rho[ik - 1]) / dr
    # Forward/backward at boundaries.
    if nr > 1:
        drho_dr[0] = (rho[1] - rho[0]) / (r[1] - r[0])
        drho_dr[-1] = (rho[-1] - rho[-2]) / (r[-1] - r[-2])

    # gradr = dr/dr . r̂
    grad = np.zeros((nr, na, 3), dtype=float)
    for d in range(3):
        grad[:, :, d] = drho_dr * omega[None, :, d]
    return grad


def _atomic_divergence(
    flux: np.ndarray,
    grid: AtomicRadialGrid,
) -> np.ndarray:
    """Compute the divergence of a vector field on an atomic grid.

    For the radial-spherical coordinate system:
    grad.F = (1/r^2) d(r^2 F_r)/dr + angular terms

    We approximate with just the radial part since that dominates
    near the nucleus.

    Parameters
    ----------
    flux
        ``(n_radial, n_angular, 3)`` vector field.
    grid
        Atomic radial grid.

    Returns
    -------
    div
        ``(n_radial, n_angular)`` divergence.
    """
    nr, na = flux.shape[:2]
    r = grid.r

    # Radial component: F_r = S_d F_d . w_d
    omega = grid.angular_xyz  # (n_angular, 3)
    F_r = np.sum(flux * omega[None, :, :], axis=-1)  # (n_radial, n_angular)

    # d(r^2 F_r)/dr via finite differences.
    r2_Fr = r[:, None] ** 2 * F_r
    d_r2Fr_dr = np.zeros_like(r2_Fr)
    for ik in range(1, nr - 1):
        dr = r[ik + 1] - r[ik - 1]
        if dr > 1e-14:
            d_r2Fr_dr[ik] = (r2_Fr[ik + 1] - r2_Fr[ik - 1]) / dr
    if nr > 1:
        d_r2Fr_dr[0] = (r2_Fr[1] - r2_Fr[0]) / (r[1] - r[0])
        d_r2Fr_dr[-1] = (r2_Fr[-1] - r2_Fr[-2]) / (r[-1] - r[-2])

    # Divergence = (1/r^2) d(r^2 F_r)/dr
    div = d_r2Fr_dr / (r[:, None] ** 2 + 1e-30)
    return div


# ============================================================
# GAPW JK Builder
# ============================================================


class GapwJBuilder:
    """Hartree-J builder for the M3c GAPW route.

    Composes the smooth-grid FFT Poisson J (via :class:`GpwJBuilder`)
    with the per-atom augmentation correction for all-electron accuracy.

    Parameters
    ----------
    basis, grid, system
        Standard vibe-qc periodic system + basis + FFT grid.
    lmax
        Multipole order for the r₀ compensator.
    soft_cutoff
        Exponent threshold for the softened basis.
    n_radial
        Number of radial grid points per atom.
    lebedev_order
        Lebedev angular grid order.
    smearing_alpha
        Gaussian width for r₀ compensator (auto-pick if None).
    cache_augmentation
        Cache the augmentation data (True by default).
    one_centre
        Explicit low-level density construction: ``"block"`` (default),
        ``"projector"``, or ``"analytic"``.  This builder has no HF/DFT
        context, so method-aware ``"auto"`` resolution belongs to the SCF
        drivers and is intentionally not accepted here.
    quiet
        Suppress experimental warnings.
    """

    def __init__(
        self,
        basis,
        system,
        grid: PlaneWaveGrid,
        *,
        lmax: int = _DEFAULT_LMAX,
        soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
        n_radial: int = _DEFAULT_N_RADIAL,
        lebedev_order: int = _DEFAULT_LEBEDEV_ORDER,
        smearing_alpha: Optional[float] = None,
        cache_augmentation: bool = True,
        one_centre: str = "block",
        quiet: bool = False,
    ) -> None:
        if one_centre not in ("block", "projector", "analytic"):
            raise ValueError(
                f"one_centre must be 'block', 'projector', or "
                f"'analytic'; got {one_centre!r}"
            )
        self._one_centre = one_centre
        self._basis = basis
        self._system = system
        self._grid = grid

        # Iteration-invariant full-basis chi (+ G-mesh), built once and shared
        # by the smooth-J builder, the V->AO projection in build_J, and the
        # driver-side XC collocate/project (reachable via .collocation_cache).
        self._collocation_cache = build_gpw_collocation_cache(basis, grid)

        # Smooth-grid GPW J builder.
        self._gpw = GpwJBuilder(
            basis, grid, collocation_cache=self._collocation_cache
        )

        # GAPW augmentation orchestrator.
        self._aug = GapwAugmentation(
            basis,
            system,
            grid,
            lmax=lmax,
            soft_cutoff=soft_cutoff,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
            smearing_alpha=smearing_alpha,
            quiet=quiet,
        )
        # Soft-basis chi for the soft-density collocation in build_J /
        # gapw_hartree_energy -- also iteration-invariant (the soft basis is
        # fixed for the SCF). Built only when the augmentation is active; the
        # soft path is never taken otherwise.
        self._soft_collocation_cache: Optional[GpwCollocationCache] = (
            build_gpw_collocation_cache(self._aug._soft_basis, grid)
            if getattr(self._aug, "_augmentation_active", False)
            else None
        )
        self._cache = cache_augmentation
        self._cached_J_correction: Optional[np.ndarray] = None
        self._cached_D: Optional[np.ndarray] = None
        # D-independent pieces of the analytic Hartree Fock (built lazily on
        # the first build_J call; see _ensure_fock_cache / build_J).
        self._fock_cache: Optional[list[dict]] = None
        # Projector one-centre mode (experimental): D-independent expansion
        # data built lazily on first use, plus a last-(D, E, J) memo so the
        # SCF's build_J + gapw_hartree_energy pair per iteration costs one
        # evaluation. See periodic_gapw_projector.py.
        self._projector_expansions = None
        self._projector_memo: Optional[tuple[np.ndarray, float,
                                             np.ndarray]] = None
        # Analytic fit-free one-centre mode (experimental): exact-ERI
        # augmentation data, built lazily; same memo pattern.
        self._analytic_aug = None
        self._analytic_memo: Optional[tuple[np.ndarray, float,
                                            np.ndarray]] = None
        # Experimental-gate warning retired (analytic Fock = dE_H/dD + overlap
        # double-count fixed); `quiet` kept for API compatibility.

    def _smooth_xc_generation(self, density_matrix: np.ndarray):
        """Pick the ``(basis, density_matrix, cache)`` that GENERATES the
        smooth-grid XC field for the GAPW partition.

        Mirrors :meth:`build_J`: when the augmentation is active the smooth
        XC term must be evaluated on the SOFT (pseudo) density r̃ -- *not* the
        full all-electron density -- so that the per-atom XC augmentation
        ``E_xc[r_a] - E_xc[r̃_a]`` telescopes to the correct all-electron XC.
        Using the full density here double-counts the soft core inside each
        augmentation sphere (the smooth term already holds e_xc[full], then
        the augmentation adds e_xc[hard]-e_xc[soft] on top): a ~0.3 Ha
        over-binding on He/STO-3G (LDA and GGA alike), independent of grid.

        The potential is still projected onto the FULL AO basis by the
        caller -- exactly as :meth:`build_J` does for the Hartree term -- so
        the soft density only restricts what *generates* the field, not the
        AO products it is integrated against.

        Returns ``(gen_basis, D_gen, gen_cache)``.
        """
        D = np.asarray(density_matrix, dtype=float)
        if not self._aug._augmentation_active:
            return self._basis, D, self._collocation_cache
        idx = self._aug._soft_indices
        D_soft = D[np.ix_(idx, idx)]
        return self._aug._soft_basis, D_soft, self._soft_collocation_cache

    def _projector_energy_and_fock(self, D: np.ndarray):
        """Projector one-centre Hartree energy + exact Fock (experimental).

        Delegates to :mod:`vibeqc.periodic_gapw_projector` (the
        oracle-validated multipole-consistent construction). The
        D-independent expansion data is built once per builder; a
        last-D memo makes the SCF's per-iteration build_J +
        gapw_hartree_energy pair cost one evaluation.
        """
        if (self._projector_memo is not None
                and self._projector_memo[0].shape == D.shape
                and np.array_equal(self._projector_memo[0], D)):
            return self._projector_memo[1], self._projector_memo[2]
        from .periodic_gapw_projector import (
            build_one_centre_expansions,
            projector_hartree_energy_and_fock,
        )

        aug, grid = self._aug, self._grid
        atom_grids = {ia: ad.grid for ia, ad in enumerate(aug._atom_data)}
        if self._projector_expansions is None:
            self._projector_expansions = build_one_centre_expansions(
                self._basis, aug._soft_basis, self._system, atom_grids)
        idx = np.asarray(aug._soft_indices, dtype=int)
        D_soft = D[np.ix_(idx, idx)]
        rho_tilde = collocate_density_on_grid(
            aug._soft_basis, D_soft, grid,
            cache=self._soft_collocation_cache,
        )
        soft_chi = np.asarray(self._soft_collocation_cache.chi)
        e, J = projector_hartree_energy_and_fock(
            self._projector_expansions, atom_grids, D, idx,
            rho_tilde, soft_chi, grid,
        )
        self._projector_memo = (D.copy(), e, J)
        return e, J

    def _analytic_energy_and_fock(self, D: np.ndarray):
        """Fit-free analytic-ERI Hartree energy + exact Fock.

        Delegates to :mod:`vibeqc.periodic_gapw_projector`'s
        analytic-ERI augmentation (validated 2026-07-31: H2O SCF at
        +11.5 mHa vs the molecular reference, no variational hole).
        Molecular-limit Gamma envelope: the free-space ERI difference
        assumes an isolated cluster in the box.
        """
        if (self._analytic_memo is not None
                and self._analytic_memo[0].shape == D.shape
                and np.array_equal(self._analytic_memo[0], D)):
            return self._analytic_memo[1], self._analytic_memo[2]
        from .periodic_gapw_projector import (
            analytic_hartree_energy_and_fock,
            build_analytic_augmentation,
        )

        aug, grid = self._aug, self._grid
        if self._analytic_aug is None:
            self._analytic_aug = build_analytic_augmentation(
                self._basis, aug._soft_basis, self._system, grid)
        idx = np.asarray(aug._soft_indices, dtype=int)
        D_soft = D[np.ix_(idx, idx)]
        rho_tilde = collocate_density_on_grid(
            aug._soft_basis, D_soft, grid,
            cache=self._soft_collocation_cache,
        )
        soft_chi = np.asarray(self._soft_collocation_cache.chi)
        e, J = analytic_hartree_energy_and_fock(
            self._analytic_aug, D, idx, rho_tilde, soft_chi, grid,
        )
        self._analytic_memo = (D.copy(), e, J)
        return e, J

    def _analytic_xc_correction(self, D: np.ndarray, functional):
        """XC augmentation on projector one-centre densities.

        The analytic mode's DFT path: block one-centre densities
        mis-telescope XC by Ha-scale on bond-inside-sphere systems
        (H2O/LDA +1567.5 mHa at fixed density); projector densities
        land at -70.5 mHa (2026-07-31 oracle). Returns
        ``({atom_index: dV_xc}, e_xc_aug)`` like the block
        ``compute_xc_correction(return_energy=True)``.
        """
        from .periodic_gapw_projector import (
            build_one_centre_expansions,
            projector_xc_correction,
        )

        aug = self._aug
        atom_grids = {ia: ad.grid for ia, ad in enumerate(aug._atom_data)}
        if self._projector_expansions is None:
            self._projector_expansions = build_one_centre_expansions(
                self._basis, aug._soft_basis, self._system, atom_grids)
        idx = np.asarray(aug._soft_indices, dtype=int)
        D = np.asarray(D, dtype=float)
        D_soft = D[np.ix_(idx, idx)]
        return projector_xc_correction(
            self._projector_expansions, atom_grids, D, D_soft, idx,
            functional,
        )

    def _analytic_xc_correction_polarised(
        self, D_alpha: np.ndarray, D_beta: np.ndarray, functional
    ):
        """Spin-polarised XC augmentation on projector densities.

        Open-shell counterpart of :meth:`_analytic_xc_correction`;
        ``functional`` must be the polarised (spin=2) wrapper. Returns
        ``({atom: dV_a}, {atom: dV_b}, e_xc_aug)`` like the block
        ``compute_xc_correction_polarised(return_energy=True)``.
        """
        from .periodic_gapw_projector import (
            build_one_centre_expansions,
            projector_xc_correction_polarised,
        )

        aug = self._aug
        atom_grids = {ia: ad.grid for ia, ad in enumerate(aug._atom_data)}
        if self._projector_expansions is None:
            self._projector_expansions = build_one_centre_expansions(
                self._basis, aug._soft_basis, self._system, atom_grids)
        idx = np.asarray(aug._soft_indices, dtype=int)
        D_a = np.asarray(D_alpha, dtype=float)
        D_b = np.asarray(D_beta, dtype=float)
        return projector_xc_correction_polarised(
            self._projector_expansions, atom_grids,
            D_a, D_b,
            D_a[np.ix_(idx, idx)], D_b[np.ix_(idx, idx)],
            idx, functional,
        )

    def _ensure_fock_cache(self) -> list[dict]:
        """Build (once) the D-independent pieces of the analytic Hartree Fock.

        The analytic exact Fock ``J = ∂E_H/∂D`` (see :meth:`build_J`) has
        per-atom pieces that depend only on the basis / grid / compensator
        widths, not on the density:

        * ``chi_f`` / ``chi_s`` -- the on-centre hard / soft AO values on the
          atomic grid (with the on-centre mask folded in);
        * the radial window ``win`` and combined quadrature weights ``wt``;
        * ``M[c,mu,nu] = ∂Q_lm/∂D_mu,nu`` -- the (l,m) multipole of the
          on-centre ``(chi_mu chi_nu - chĩ_mu chĩ_nu)`` product;
        * ``grid_unit[c]`` / ``rad_unit[c]`` -- the FFT-grid and atomic-grid
          densities of the *unit* compensator (Q=1 on component c), used to
          project the per-component compensator response onto V_smooth /
          V_soft.

        Recomputing these every SCF iteration (in particular the per-(l,m)
        ``density_on_grid`` collocation) is what made the naive blueprint too
        slow for the suite; caching them leaves only the D-dependent radial
        Poisson solves + the FFT smooth solve per iteration.
        """
        if self._fock_cache is not None:
            return self._fock_cache

        aug, grid = self._aug, self._grid
        lmax = aug._lmax
        n_comp = (lmax + 1) ** 2
        # l(c) = floor(sqrt(c)) for the (l,m) flattening used by the radial
        # spherical-harmonic / multipole convention.
        lof = np.array(
            [int(np.floor(np.sqrt(c))) for c in range(n_comp)], dtype=int
        )
        # A zero-density compensator fixes the (D-independent) positions/alpha
        # the unit compensators must share with _build_compensator.
        n_full = self._basis.nbasis
        comp0 = aug._build_compensator(np.zeros((n_full, n_full)))
        saved_comp = aug._compensator

        cache: list[dict] = []
        for ad in aug._atom_data:
            g_r = ad.grid
            r = np.asarray(g_r.r)
            w_r, w_a = np.asarray(g_r.w_r), np.asarray(g_r.w_a)
            na = g_r.n_angular
            # Partition-of-unity weight (replaces the bare radial window so
            # overlapping spheres do not double-count; reduces to the window
            # for an isolated atom). Geometry-only -> cached on the aug.
            part = aug._partition_weight(ad)
            wt = g_r.combined_weights()
            partf = part.ravel()
            wtf = wt.ravel()
            chi_f = ad.chi_a * ad.on_center_full[None, :]
            chi_s = ad.chi_tilde * ad.on_center_soft[None, :]

            # M_{c,mu,nu} = (l,m) multipole of the on-centre (chi chi - chĩ chĩ).
            S = g_r.evaluate_real_spherical_harmonics(lmax)
            Wc = np.empty((n_comp, g_r.n_radial * na))
            for c in range(n_comp):
                Wc[c] = (
                    S[c][None, :]
                    * (r ** lof[c])[:, None]
                    * (w_r[:, None] * w_a[None, :])
                ).ravel()
            M = np.einsum("cg,gm,gn->cmn", Wc, chi_f, chi_f, optimize=True)
            M[:, aug._soft_indices[:, None], aug._soft_indices[None, :]] -= (
                np.einsum("cg,gm,gn->cmn", Wc, chi_s, chi_s, optimize=True)
            )

            # Unit-compensator densities (Q=1 on one component) on the FFT grid
            # and on this atom's radial grid -- the response basis.
            grid_unit_list: list[np.ndarray] = []
            rad_unit_list: list[np.ndarray] = []
            for c in range(n_comp):
                Qu = np.zeros((comp0.n_atoms, n_comp))
                Qu[ad.atom_idx, c] = 1.0
                cu = GaussianMultipoleCompensator(
                    positions=comp0.positions,
                    alpha=comp0.alpha,
                    Q=Qu,
                    lmax=lmax,
                )
                grid_unit_list.append(np.asarray(cu.density_on_grid(grid)))
                aug._compensator = cu
                rad_unit_list.append(
                    np.asarray(aug.compensator_density_on_atomic_grid(ad))
                )
            cache.append(
                {
                    "chi_f": chi_f,
                    "chi_s": chi_s,
                    "part": part,
                    "wt": wt,
                    "partf": partf,
                    "wtf": wtf,
                    "M": M,
                    "grid_unit": np.asarray(grid_unit_list),
                    "rad_unit": np.asarray(rad_unit_list),
                    "n_comp": n_comp,
                }
            )

        aug._compensator = saved_comp
        self._fock_cache = cache
        return cache

    def build_J(self, density_matrix: np.ndarray) -> np.ndarray:
        """Build the all-electron GAPW Hartree-J matrix ``J = ∂E_H/∂D``.

        This is the **exact analytic** GAPW Hartree Fock: the derivative of
        :meth:`gapw_hartree_energy` with respect to the density matrix, so
        ``½ tr(D·J) = E_H`` holds and the SCF aufbau finds the
        core-occupied state. (The earlier
        ``J_smooth(full) + Σ_a ∫χχ(V_hard − V_soft)`` form was *not*
        ``∂E_H/∂D``: it omitted the soft-basis projection of the smooth term
        and the D-dependent compensator response, pushing a partially-occupied
        core's tight 1s above the valence -- O collapsed to −35 Ha with an
        empty core. See ``handovers/HANDOVER_GAPW_PRODUCTION.md`` § milestone 3
        and ``examples/regression/gapw_parity/milestone3_analytic_fock.py``.)

        Since ``E_H`` is quadratic in D, the Fock is ``∫ (∂ρ/∂D_μν) V[ρ]`` of
        every density piece:

            J_μν = embed(∫_FFT χ̃_μ χ̃_ν V_smooth)                   # (A) smooth, SOFT basis
                 + Σ_a p_a·∫_rad (χ_oc χ_oc) V_hard,a               # (B) hard, full, on-centre
                 − Σ_a p_a·embed(∫_rad (χ̃_oc χ̃_oc) V_soft,a)       # (C) soft, soft basis
                 + Σ_{a,lm} M_{lm,a,μν}(⟨g_lm|V_smooth⟩−⟨g_lm|V_soft⟩)  # (D) ρ₀ response

        where ``p_a`` is the partition-of-unity weight (``_partition_weight``;
        no double-count where augmentation spheres overlap, reducing to the
        bare radial window for an isolated atom) and ``V_soft,a`` is built from
        atom a's OWN compensator only (the global compensator lives on the
        smooth FFT grid).

        where ``M_{lm,a,μν} = ∂Q_lm,a/∂D_μν`` is the (l,m) multipole of the
        on-centre ``(χ_μχ_ν − χ̃_μχ̃_ν)`` product (term (D) -- the compensator
        ρ₀ is D-dependent, Q_lm[D] -- is what repairs the core 1s).

        Parameters
        ----------
        density_matrix
            ``(n_basis, n_basis)`` AO density matrix.

        Returns
        -------
        J
            ``(n_basis, n_basis)`` symmetrised all-electron J matrix.
        """
        D = np.asarray(density_matrix, dtype=float)

        # If the augmentation is inactive (soft basis == full basis, no
        # pruned core), there is no hard/soft split: the plain GPW J on
        # the full density is already the right Hartree matrix.
        if not self._aug._augmentation_active:
            return self._gpw.build_J(D)

        if self._one_centre == "projector":
            _e, J = self._projector_energy_and_fock(D)
            return J
        if self._one_centre == "analytic":
            _e, J = self._analytic_energy_and_fock(D)
            return J

        aug, grid = self._aug, self._grid
        n = self._basis.nbasis
        idx = aug._soft_indices
        dV = grid.voxel_volume_bohr3
        cache = self._ensure_fock_cache()

        # Smooth Hartree potential V_H[ρ̃ + ρ₀] on the FFT grid: the soft
        # density ρ̃ (the full core cannot be represented on the coarse grid)
        # plus the hard-minus-soft compensator ρ₀ restoring the lost core
        # multipoles. The compensator is rebuilt from the current D.
        D_soft = D[np.ix_(idx, idx)]
        comp = aug._build_compensator(D)
        rho_tilde = collocate_density_on_grid(
            aug._soft_basis, D_soft, grid,
            cache=self._soft_collocation_cache,
        )
        V_smooth = _core.solve_poisson_coulomb(
            rho_tilde + comp.density_on_grid(grid), grid.lattice_bohr
        )

        # (A) smooth term on the SOFT basis (∂ñ/∂D uses the soft AOs), embedded
        # into the full-dimensional Fock.
        J = np.zeros((n, n))
        J[np.ix_(idx, idx)] = project_potential_to_ao(
            aug._soft_basis, V_smooth, grid,
            cache=self._soft_collocation_cache,
        )

        for ad, cd in zip(aug._atom_data, cache):
            chi_f, chi_s = cd["chi_f"], cd["chi_s"]
            part, wt, partf, wtf, M = (
                cd["part"], cd["wt"], cd["partf"], cd["wtf"], cd["M"]
            )
            rho_a = aug._compute_atomic_density(D, ad, use_soft=False)
            rho_t = aug._compute_atomic_density(D, ad, use_soft=True)
            # Own-atom compensator only (the global compensator is on the
            # smooth grid); see compensator_density_on_atomic_grid.
            rho0a = aug.compensator_density_on_atomic_grid(ad, own_atom_only=True)
            V_hard = solve_poisson_radial(rho_a, ad.grid)
            V_soft = solve_poisson_radial(rho_t + rho0a, ad.grid)

            # (B) on-centre hard, full basis; (C) on-centre soft, soft basis.
            # ``partf`` is the partition-of-unity weight (no double-count in
            # overlapping spheres); reduces to the radial window if isolated.
            J += np.einsum(
                "gm,gn,g->mn", chi_f, chi_f, V_hard.ravel() * partf * wtf,
                optimize=True,
            )
            J[np.ix_(idx, idx)] -= np.einsum(
                "gm,gn,g->mn", chi_s, chi_s, V_soft.ravel() * partf * wtf,
                optimize=True,
            )

            # (D) compensator ρ₀ response: project the cached unit-compensator
            # densities onto V_smooth (grid) and V_soft (radial), contract with
            # the cached ∂Q/∂D multipole tensor M.
            grid_unit, rad_unit = cd["grid_unit"], cd["rad_unit"]
            v_sm = np.einsum("cg,g->c", grid_unit.reshape(cd["n_comp"], -1),
                             V_smooth.ravel(), optimize=True) * dV
            v_so = np.einsum(
                "cra,ra,ra->c", rad_unit, V_soft * part, wt,
                optimize=True,
            )
            J += np.einsum("c,cmn->mn", v_sm - v_so, M, optimize=True)

        return 0.5 * (J + J.T)

    def build_J_bloch(
        self,
        density_matrix: np.ndarray,
        rho_tilde_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """GAPW Hartree pieces for the multi-k Bloch (compact-cell) regime.

        :meth:`build_J` returns one Gamma AO matrix, which is only valid
        when the Gamma-folded density is the true periodic density. On a
        compact cell the smooth term must be built from the **Bloch**
        density and projected **per k**; see the fold-charge guard in
        :func:`run_periodic_rks_gapw_multi_k` for the measurement that
        makes this necessary (LiH rocksalt: 9.82 e for a 4-electron cell).

        The split follows the term labels in :meth:`build_J`:

        * **(A) smooth** is a local potential. It is returned as the grid
          potential ``V_smooth`` so the caller can project it onto the
          SOFT Bloch AOs at each k (``project_potential_to_bloch_ao``)
          and embed the result in the soft block.
        * **(B), (C), (D)** are one-centre objects living inside the
          atomic augmentation spheres. Those regions are non-overlapping
          by construction (the GAPW premise; Lippert-Hutter-Parrinello
          1999 Eq. 15), so the image pairs that reach into sphere ``a``
          are the same-cell ones and these blocks are k-independent.
          They are built from ``D(g=0) = sum_k w_k D(k)`` -- the
          on-site block, which is exactly the right object for a
          one-centre density -- and returned as one real matrix added at
          every k.

        Parameters
        ----------
        density_matrix
            ``D(g=0)``, the BZ-weighted real density matrix.
        rho_tilde_grid
            The SOFT density already collocated on the FFT grid in the
            caller's regime (Bloch sum for compact cells). The caller
            owns this because the multi-k driver already builds it for
            the XC path.

        Returns
        -------
        (V_smooth, J_local, e_hartree)
            ``e_hartree`` is the density-functional GAPW Hartree energy
            (``1/2 int (rho~ + rho_0) V[rho~ + rho_0]`` plus the per-atom
            augmentation), NOT ``1/2 tr(D.J)`` -- the latter contracts
            the full density against the soft potential and is Ha-scale
            wrong for a deep core (see :meth:`gapw_hartree_energy`).
        """
        if not self._aug._augmentation_active:
            raise NotImplementedError(
                "GapwJBuilder.build_J_bloch: no augmentation is active "
                "(soft basis == full basis); use the plain GPW multi-k "
                "route, which already handles the Bloch regime."
            )
        if self._one_centre != "block":
            raise NotImplementedError(
                "GapwJBuilder.build_J_bloch: only one_centre='block' is "
                f"wired for the multi-k Bloch regime; got "
                f"{self._one_centre!r}."
            )

        aug, grid = self._aug, self._grid
        n = self._basis.nbasis
        idx = aug._soft_indices
        dV = grid.voxel_volume_bohr3
        cache = self._ensure_fock_cache()

        D = np.asarray(density_matrix, dtype=float)
        rho_tilde = np.asarray(rho_tilde_grid, dtype=float)
        if rho_tilde.shape != grid.shape:
            raise ValueError(
                f"build_J_bloch: rho_tilde_grid shape {rho_tilde.shape} "
                f"does not match the grid {grid.shape}"
            )

        # Compensator from the one-centre (same-cell) multipoles.
        comp = aug._build_compensator(D)
        rho_smooth = rho_tilde + comp.density_on_grid(grid)
        V_smooth = _core.solve_poisson_coulomb(
            rho_smooth, grid.lattice_bohr
        )
        e_smooth = 0.5 * float(np.sum(rho_smooth * V_smooth)) * dV

        # (B) + (C) + (D): one-centre, k-independent.
        J_local = np.zeros((n, n))
        for ad, cd in zip(aug._atom_data, cache):
            chi_f, chi_s = cd["chi_f"], cd["chi_s"]
            part, wt, partf, wtf, M = (
                cd["part"], cd["wt"], cd["partf"], cd["wtf"], cd["M"]
            )
            rho_a = aug._compute_atomic_density(D, ad, use_soft=False)
            rho_t = aug._compute_atomic_density(D, ad, use_soft=True)
            rho0a = aug.compensator_density_on_atomic_grid(
                ad, own_atom_only=True
            )
            V_hard = solve_poisson_radial(rho_a, ad.grid)
            V_soft = solve_poisson_radial(rho_t + rho0a, ad.grid)

            J_local += np.einsum(
                "gm,gn,g->mn", chi_f, chi_f,
                V_hard.ravel() * partf * wtf, optimize=True,
            )
            J_local[np.ix_(idx, idx)] -= np.einsum(
                "gm,gn,g->mn", chi_s, chi_s,
                V_soft.ravel() * partf * wtf, optimize=True,
            )

            grid_unit, rad_unit = cd["grid_unit"], cd["rad_unit"]
            v_sm = np.einsum(
                "cg,g->c", grid_unit.reshape(cd["n_comp"], -1),
                V_smooth.ravel(), optimize=True,
            ) * dV
            v_so = np.einsum(
                "cra,ra,ra->c", rad_unit, V_soft * part, wt, optimize=True,
            )
            J_local += np.einsum("c,cmn->mn", v_sm - v_so, M, optimize=True)

        J_local = 0.5 * (J_local + J_local.T)

        # Density-functional Hartree energy: smooth + per-atom augmentation.
        e_h_aug, _ = aug.compute_augmentation_energy(D, functional=None)
        return V_smooth, J_local, e_smooth + float(e_h_aug)

    def build_K(self, density_matrix: np.ndarray) -> np.ndarray:
        """Build the exchange matrix via the existing K builder.

        For hybrids, the K matrix uses the same Γ-only direct
        AO-image-summed exchange as the GPW route.
        """
        D = np.asarray(density_matrix, dtype=float)
        lo = _core.LatticeSumOptions()
        lo.cutoff_bohr = 25.0
        jk = _core.build_jk_gamma_molecular_limit(
            self._basis,
            self._system,
            lo,
            D,
        )
        return np.asarray(jk.K)

    def hartree_energy(self, density_matrix: np.ndarray) -> float:
        """Return the all-electron Hartree energy ``1/2 tr(D.J)``.

        .. warning::
           For the GAPW partition this is **not** the correct Hartree
           energy -- it contracts the *full* density against the *soft*
           potential (a Ha-scale error for core atoms). Use
           :meth:`gapw_hartree_energy` for the energy; ``1/2 tr(D.J)`` is
           retained only as a diagnostic / Fock-trace quantity.
        """
        J = self.build_J(density_matrix)
        return 0.5 * float(np.einsum("ij,ij->", density_matrix, J))

    def gapw_hartree_energy(self, density_matrix: np.ndarray) -> float:
        """GAPW Hartree energy in the density-functional form.

        The Hartree energy of the GAPW partition is **not** ``1/2 tr(D.J)``:
        the smooth term contracts the soft+compensator density against its
        own potential, and the augmentation is the pointwise radial
        hard/soft difference (Krack & Parrinello, *Phys. Chem. Chem. Phys.*
        2, 2105 (2000), Eq. 9):

            E_H = 1/2 ∫ (r̃+r₀) V_H[r̃+r₀]                  (smooth, FFT grid)
                + S_a [ 1/2∫ r_a V_H[r_a]
                        - 1/2∫ (r̃_a+r₀,a) V_H[r̃_a+r₀,a] ]  (augmentation)

        ``1/2 tr(D.J)`` would contract the *full* density r against the soft
        potential V_H[r̃+r₀] -- wrong by 1/2∫(r-r̃-r₀)V_H[r̃+r₀], which is
        Ha-scale for a deep core. The smooth term uses the zero-mean
        Poisson gauge (G=0 dropped); it is gauge-consistent with the Ewald
        V_ne / E_nn because r̃+r₀ carries the full electron count N, exactly
        as the plain-GPW full density does.
        """
        D = np.asarray(density_matrix, dtype=float)
        if not self._aug._augmentation_active:
            J = self._gpw.build_J(D)
            return 0.5 * float(np.einsum("ij,ij->", D, J))
        if self._one_centre == "projector":
            e, _J = self._projector_energy_and_fock(D)
            return e
        if self._one_centre == "analytic":
            e, _J = self._analytic_energy_and_fock(D)
            return e
        idx = self._aug._soft_indices
        D_soft = D[np.ix_(idx, idx)]
        rho_tilde = collocate_density_on_grid(
            self._aug._soft_basis, D_soft, self._grid,
            cache=self._soft_collocation_cache,
        )
        comp = self._aug._build_compensator(D)
        rho_sm = rho_tilde + comp.density_on_grid(self._grid)
        V_sm = _core.solve_poisson_coulomb(rho_sm, self._grid.lattice_bohr)
        dV = self._grid.voxel_volume_bohr3
        e_smooth = 0.5 * float(np.sum(rho_sm * V_sm)) * dV
        e_h_aug, _ = self._aug.compute_augmentation_energy(D, functional=None)
        return e_smooth + e_h_aug


# ============================================================
# GAPW SCF result and entry points
# ============================================================


@dataclass(frozen=True)
class GapwScfResult:
    """Result of a converged single-point GAPW periodic SCF.

    Same shape as :class:`GpwScfResult` but with the all-electron
    GAPW energies from the augmentation correction.

    Attributes
    ----------
    energy
        Total all-electron periodic GAPW energy (Hartree).
    breakdown
        :class:`GpwEnergyBreakdown` evaluated at the converged
        density (includes the GAPW correction).
    density
        Converged AO density matrix.
    mo_coeffs
        MO coefficient matrix.
    mo_energies
        MO energies.
    converged
        SCF convergence flag.
    n_iter
        Number of SCF iterations.
    grid
        The FFT grid used.
    gapw_correction
        The augmentation correction energy (Hartree).
    one_centre
        Resolved one-centre construction used by the SCF.
    fock, overlap
        Post-DIIS Fock and Gamma overlap used for the returned eigenpairs.
    molecular_limit_declared
        Whether the caller declared the isolated single-Gamma envelope (or
        selected explicit analytic mode, which is the expert declaration).
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density: np.ndarray
    mo_coeffs: np.ndarray
    mo_energies: np.ndarray
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    gapw_correction: float = 0.0
    e_dft_plus_u: float = 0.0
    dft_plus_u_sites: tuple = ()
    scf_trace: tuple = ()
    smearing_temperature: float = 0.0
    smearing_entropy: float = 0.0
    fermi_level: float = 0.0
    occupations: tuple = ()
    # The converged (post-DIIS) Fock diagonalized to produce
    # ``mo_coeffs``/``mo_energies`` and the Gamma lattice overlap. The
    # runner adapter surfaces these directly; without them it falls back
    # to reconstructing an HF-shaped ``Hcore + J - K/2`` whose J is the
    # *un-augmented* GpwJBuilder J -- not the operator this SCF
    # diagonalized.
    fock: Optional[np.ndarray] = None
    overlap: Optional[np.ndarray] = None
    one_centre: str = "block"
    molecular_limit_declared: bool = False

    @property
    def free_energy(self) -> float:
        """Mermin free energy for the returned electronic state."""

        return float(self.energy) - float(self.smearing_temperature) * float(
            self.smearing_entropy
        )

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _resolve_one_centre_mode(
    one_centre: str,
    *,
    functional: Optional[str],
    molecular_limit: bool,
) -> str:
    """Resolve the public GAPW one-centre policy to a builder mode.

    The fit-free analytic-ERI Hartree construction is the validated HF
    default inside a caller-declared molecular-limit cell. DFT keeps the
    historical block construction because the
    analytic mode's fitted off-centre XC densities are not yet variationally
    safe for multi-atom cells.  The low-level :class:`GapwJBuilder` retains
    its explicit ``"block"`` default because it has no method context.
    """
    mode = str(one_centre).strip().lower()
    if mode == "auto":
        if functional is not None:
            return "block"
        if not molecular_limit:
            raise NotImplementedError(
                "GAPW HF one_centre='auto' requires molecular_limit=True. "
                "The fit-free analytic augmentation is validated for an "
                "isolated molecule or atom in a single-Gamma box, while "
                "dense cells need periodic image sums that are not yet "
                "implemented; silently falling back to the legacy block "
                "functional would restore the known H2O wrong answer. Use "
                "GDF/BIPOLE for compact crystals, declare "
                "molecular_limit=True for a vacuum-padded cell, or select "
                "an explicit diagnostic one_centre mode."
            )
        return "analytic"
    if mode not in ("block", "projector", "analytic"):
        raise ValueError(
            "one_centre must be 'auto', 'block', 'projector', or "
            f"'analytic'; got {one_centre!r}"
        )
    return mode


def _preflight_analytic_eri_memory(
    system,
    basis,
    grid: PlaneWaveGrid,
    *,
    functional: Optional[str],
    soft_cutoff: float,
    lmax: int,
    n_radial: int,
    lebedev_order: int,
    open_shell: bool,
    memory_override: bool,
) -> None:
    """Guard the fit-free analytic route before allocating dense ERIs.

    The analytic one-centre builder retains full hard and soft four-index
    tensors.  The normal periodic runner has a route-level preflight, but the
    public standalone GAPW drivers must enforce the same contract.  Hydrogen-
    only and otherwise unpruned bases have no hard/soft augmentation and exit
    before either tensor is built, so they deliberately skip this extra gate.
    """
    soft = softened_basis(basis, system, soft_cutoff=soft_cutoff)
    full_n_prim = sum(len(sh.exponents) for sh in basis.shells())
    soft_n_prim = sum(len(sh.exponents) for sh in soft.shells())
    if soft_n_prim == full_n_prim:
        return

    from .memory import check_memory, estimate_periodic_gpw_gapw

    functional_kind: Optional[str] = None
    if functional is not None:
        func = _core.Functional(str(functional), 2 if open_shell else 1)
        kind = getattr(getattr(func, "kind", None), "name", None)
        functional_kind = str(
            kind if kind is not None else getattr(func, "kind", "")
        )

    estimate = estimate_periodic_gpw_gapw(
        n_basis=int(basis.nbasis),
        n_soft_basis=int(soft.nbasis),
        n_grid_points=int(grid.n_points),
        route="gapw",
        functional_kind=functional_kind,
        open_shell=open_shell,
        n_kpoints=1,
        n_atoms=len(system.unit_cell),
        augmentation_active=True,
        analytic_eri_one_centre=True,
        lmax=lmax,
        n_radial=n_radial,
        lebedev_order=lebedev_order,
    )
    check_memory(estimate, allow_exceed=memory_override)


def run_periodic_rhf_gapw(
    system,
    basis,
    *,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 50,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[np.ndarray] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    dispersion: Optional[str] = None,
    dispersion_functional: Optional[str] = None,
    smearing_temperature: float = 0.0,
    smearing_method: str = "fermi-dirac",
    dft_plus_u_sites=None,
    lmax: int = _DEFAULT_LMAX,
    soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
    n_radial: int = _DEFAULT_N_RADIAL,
    lebedev_order: int = _DEFAULT_LEBEDEV_ORDER,
    one_centre: str = "auto",
    molecular_limit: bool = False,
    memory_override: bool = False,
    quiet: bool = False,
    external_xc_grid_options=None,
    external_xc_image_radius_bohr: float = 10.0,
    external_xc_lattice_options=None,
) -> GapwScfResult:
    """Closed-shell GAPW RHF SCF on a periodic cell.

    ``one_centre="auto"`` selects the fit-free analytic-ERI Hartree
    augmentation for HF when ``molecular_limit=True`` declares an isolated
    single-Gamma cell, and retains the block construction for DFT.  The
    analytic construction
    (:mod:`vibeqc.periodic_gapw_projector`): the exact hard/soft ERI
    difference plus charge-conserving s-compensators, every term an
    exact contraction or bounded-kernel quadrature. It is the HF default
    because the historical block construction omits off-centre AO tails from
    each atom's one-centre density. Validated
    2026-07-31: H2O/STO-3G converges to +11.5 mHa vs the molecular
    reference where the legacy block augmentation is -8.86 Ha wrong;
    single atoms Ne/O/He match their fixed-density identity accuracy.
    Molecular-limit Gamma envelope (isolated cluster in the box).

    ``one_centre="projector"`` (experimental) uses the fitted
    projector construction instead; single-atom cells only (off-centre
    fit errors are variationally exploitable - see the handover) and
    HF-only.

    DFT on the analytic mode supports SINGLE-ATOM cells, where the
    XC augmentation runs on exact on-centre densities and lands at
    HF-quality accuracy across bases (Ne LDA +15.1 / +18.1 / +22.7
    mHa at 6-31G / STO-3G / def2-SVP vs molecular RKS). Multi-atom
    DFT fails closed: the fitted off-centre one-centre densities'
    variational softness grows with basis richness (LiH LDA -10.6 ->
    -105 mHa from STO-3G to def2-SVP; H2O up to -21 Ha), so no basis
    calibration can bound the class - a variationally safe XC
    formulation for fitted content is an open research item.

    Same interface and conventions as :func:`run_periodic_rhf_gpw`
    but uses the GAPW J builder with per-atom augmentation correction
    for all-electron accuracy.

    Parameters
    ----------
    system, basis
        Standard periodic system + basis.
    grid, cutoff_ha
        FFT grid or cutoff.
    max_iter, conv_tol_energy, conv_tol_density
        SCF convergence parameters.
    damping
        Linear density mixing damping.
    initial_density
        Initial density matrix (None -> Hcore guess).
    v_ne_convention
        V_ne convention: "ewald" or "smeared_erfc".
    smearing_alpha
        Smearing exponent for the smeared V_ne (bohr⁻¹).
    functional
        XC functional name (None -> HF exchange only).
    use_diis, diis_subspace_size, diis_start_iter
        DIIS acceleration parameters.
    dispersion, dispersion_functional
        D3-BJ dispersion correction (requires the ``dftd3`` python
        bindings; raises ``NotImplementedError`` when missing).
    smearing_temperature, smearing_method
        Fermi-Dirac smearing parameters.
    dft_plus_u_sites
        DFT+U Hubbard sites.
    lmax
        Multipole order for the r₀ compensator.
    soft_cutoff
        Exponent threshold for the softened basis (bohr⁻^2).
    n_radial
        Number of radial points per atomic grid.
    lebedev_order
        Lebedev angular grid order.
    one_centre
        Method-aware policy. With ``molecular_limit=True``, ``"auto"``
        (default) selects ``"analytic"`` for HF; without that declaration HF
        fails closed. DFT ``"auto"`` selects ``"block"``. Explicit
        ``"block"`` retains the historical AO-block construction and
        ``"projector"`` is the fitted single-atom HF diagnostic mode.
    molecular_limit
        Declare that the cell is a vacuum-padded isolated molecule or atom at
        one Gamma point. Required for automatic HF selection because no
        geometry-only classifier can distinguish that envelope reliably from
        an arbitrary dense-crystal supercell. Explicit ``"analytic"`` remains
        the expert opt-in spelling of the same declaration.
    memory_override
        Permit the fit-free analytic route to exceed the live-memory
        preflight. The retained hard/soft four-index ERI tensors are charged
        before allocation.
    quiet
        Suppress experimental warnings.

    Returns
    -------
    result
        :class:`GapwScfResult` with the converged all-electron energy.
    """
    input_restart_supplied = initial_density is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver="run_periodic_rhf_gapw",
        supported=periodic_guess_capabilities('gapw', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
    )
    if initial_density is not None:
        from .guess_read import _real_density
        initial_density = _real_density(initial_density)


    # Experimental-gate warning retired (correctness bugs fixed 2026-06-26);
    # remaining accuracy caveats are in docs/user_guide/gapw.md.
    if system.dim != 3:
        raise ValueError(f"run_periodic_rhf_gapw: dim == 3 only; got dim={system.dim}")
    if functional is not None:
        requested_func = _core.Functional(functional, 1)
        if (
            bool(getattr(requested_func, "is_external", False))
            and float(system.charge) != 0.0
        ):
            raise NotImplementedError(
                "run_periodic_rhf_gapw: charged-cell external XC is not "
                "supported because the GAPW Coulomb background and electron-"
                "count convention are not validated; use a neutral cell"
            )
        if bool(getattr(requested_func, "is_external", False)):
            from .pbc_bipole_common import reject_bipole_ecp_options
            from .periodic_external_xc import (
                _require_zero_temperature_external_xc,
            )

            _require_zero_temperature_external_xc(
                smearing_temperature,
                where="run_periodic_rhf_gapw",
            )

            try:
                reject_bipole_ecp_options(
                    object(),
                    driver="run_periodic_rhf_gapw external XC",
                    basis=basis,
                    system=system,
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    "run_periodic_rhf_gapw: ECP-bearing bases are not "
                    "implemented with external XC; use an all-electron basis"
                ) from exc
    if not isinstance(molecular_limit, (bool, np.bool_)):
        raise TypeError(
            "run_periodic_rhf_gapw: molecular_limit must be bool; "
            f"got {type(molecular_limit).__name__}."
        )

    requested_one_centre = str(one_centre).strip().lower()
    one_centre = _resolve_one_centre_mode(
        one_centre,
        functional=functional,
        molecular_limit=bool(molecular_limit),
    )
    molecular_limit_declared = bool(
        molecular_limit or requested_one_centre == "analytic"
    )

    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_periodic_rhf_gapw: cell has {n_elec} electrons (odd); "
            f"closed-shell needs an even count."
        )
    n_occ = n_elec // 2

    smearing_T = float(smearing_temperature)
    if smearing_T < 0.0:
        raise ValueError(f"smearing_temperature must be >= 0; got {smearing_T}")
    if smearing_T > 0.0:
        from .smearing import SmearingOptions as _SmearingOptions

        smearing_opts = _SmearingOptions(
            temperature=smearing_T,
            flavor=str(smearing_method),
        )
    else:
        smearing_opts = None

    # Build grid if needed.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    if one_centre == "analytic":
        _preflight_analytic_eri_memory(
            system,
            basis,
            grid,
            functional=functional,
            soft_cutoff=soft_cutoff,
            lmax=lmax,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
            open_shell=False,
            memory_override=memory_override,
        )

    # One-electron integrals.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(
        basis,
        system,
        v_ne_convention,
        smearing_alpha,
        grid,
    )
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne

    # Ewald nuclear repulsion.
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    if one_centre == "projector" and functional is not None:
        raise NotImplementedError(
            "run_periodic_rhf_gapw: one_centre='projector' supports "
            "HF only (the XC augmentation still uses the "
            "block-restricted one-centre densities, and the projector "
            "mode itself is single-atom-only). Drop functional= "
            "or use one_centre='block'."
        )
    # Analytic-mode DFT runs the XC augmentation on projector one-centre
    # densities (see _analytic_xc_correction). That fixes the block XC
    # Ha-scale mis-telescoping, but the fitted densities' XC is still
    # variationally soft at MINIMAL bases: H2O/STO-3G LDA converges
    # -545 mHa below molecular (fixed-density oracle: -70) while
    # H2O/6-31G lands +72 mHa - the drift shrinks ~8x with a
    # double-zeta basis (KP2000 report sub-mHa at 6-31G*). Until that
    # is calibrated across real bases, the >=2-neighbour pruned-core
    # DFT class stays fail-closed. Meta-GGA is additionally rejected
    # inside the projector XC.
    if (one_centre == "analytic" and functional is not None
            and len(list(system.unit_cell)) > 1):
        # DFT on the analytic route is validated for single-atom cells
        # only: there the XC augmentation uses exact on-centre
        # densities and lands at HF-quality accuracy across bases
        # (Ne LDA +15.1 / +18.1 / +22.7 mHa at 6-31G / STO-3G /
        # def2-SVP after the soft-block V_xc projection fix). Any
        # multi-atom cell involves OFF-CENTRE fitted one-centre
        # densities in the XC telescoping, and their variational
        # softness GROWS with basis richness (measured 2026-08-01,
        # LDA vs molecular RKS: LiH -10.6 / -54.1 / -105.0 mHa and
        # H2O -545 / +72 / -21106 at STO-3G / 6-31G / def2-SVP).
        # A basis-quality calibration cannot bound that class; a
        # variationally safe XC formulation for fitted off-centre
        # content is an open research item. HF on this route is
        # unrestricted and validated (H2O +7.4 mHa at 6-31G).
        raise NotImplementedError(
            "run_periodic_rhf_gapw: one_centre='analytic' with "
            f"functional={functional!r} supports single-atom cells "
            "only. Multi-atom XC telescoping runs on fitted "
            "off-centre one-centre densities whose variational "
            "softness grows with basis richness (LiH LDA -10.6 -> "
            "-105 mHa from STO-3G to def2-SVP; H2O up to -21 Ha). "
            "Use jk_method='gdf'/'bipole' for DFT on molecules, or "
            "HF on this route."
        )
    if one_centre == "projector" and len(list(system.unit_cell)) > 1:
        raise NotImplementedError(
            "run_periodic_rhf_gapw: one_centre='projector' supports "
            "single-atom cells only for now. Multi-atom cells need "
            "off-centre one-centre fits, and the SCF exploits their "
            "fit-error directions variationally (H2O/STO-3G converges "
            "-3.67 Ha below the molecular reference even though the "
            "functional is accurate to +33 mHa at the physical "
            "density; see HANDOVER_GAPW_PRODUCTION.md 2026-07-30). "
            "Use one_centre='block' or jk_method='gdf'."
        )

    # GAPW J builder.
    gapw_builder = GapwJBuilder(
        basis,
        system,
        grid,
        lmax=lmax,
        soft_cutoff=soft_cutoff,
        n_radial=n_radial,
        lebedev_order=lebedev_order,
        one_centre=one_centre,
        quiet=quiet,
    )

    # Density-mode guesses use the shared periodic adapter. HCORE returns
    # ``None`` and retains the route's historical diagonalisation. Explicit
    # restart density always takes precedence.
    if initial_density is None:
        from .guess import initial_density_closed_shell

        initial_density = initial_density_closed_shell(
            system.unit_cell_molecule(),
            basis,
            n_occ,
            (_core.InitialGuess.SAD if initial_guess == _core.InitialGuess.PATOM
             else initial_guess),
            is_periodic=True,
            periodic_system=system,
            lattice_opts=_core.LatticeSumOptions(),
            overlap=S,
        )
    if initial_density is None:
        s_eigs, U = _eigh_safe(S)
        S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T
        e, C_orth = _eigh_safe(S_half_inv @ Hcore @ S_half_inv)
        C = S_half_inv @ C_orth
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    else:
        D = np.asarray(initial_density, dtype=float).copy()
        if D.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density shape {D.shape} doesn't match basis "
                f"({basis.nbasis} functions)"
            )

    from .guess import normalize_density_guess
    D = normalize_density_guess(D, S, 2 * n_occ)

    if initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_spin_density_step
        from .periodic_gapw_open_shell import _k_per_spin
        seed_options = _core.LatticeSumOptions()
        seed_options.cutoff_bohr = 25.0
        alpha, beta = patom_spin_density_step(
            D * 0.5, D * 0.5, Hcore, S, n_occ, n_occ,
            gapw_builder.build_J,
            lambda d: _k_per_spin(basis, system, d, seed_options),
        )
        D = alpha + beta

    # Pre-build orthogonalizer.
    s_eigs, U = _eigh_safe(S)
    S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T

    # Optional functional.
    if functional is not None:
        func = _core.Functional(functional, 1)
        # Full-range-only K on this route (see run_periodic_rhf_gpw).
        reject_unscreened_range_separated(
            func, where="run_periodic_rhf_gapw"
        )
        ex_frac = float(func.hf_exchange_fraction)
        # Meta-GGA is supported on this Γ closed-shell GAPW route: the
        # smooth τ Fock term comes from the smooth-grid XC build below
        # (with its own von Weizsacker floor + fail-closed stiffness guard,
        # so TPSS-class functionals still fail closed there), and the per-atom
        # τ augmentation telescopes the core t (compute_xc_correction, the
        # _project_atomic_vtau channel).
    else:
        func = None
        ex_frac = 1.0

    external_xc = None
    if func is not None and bool(getattr(func, "is_external", False)):
        from .periodic_external_xc import PeriodicExternalXC

        if external_xc_lattice_options is None:
            external_lattice_options = _core.LatticeSumOptions()
            external_lattice_options.cutoff_bohr = 25.0
        else:
            external_lattice_options = external_xc_lattice_options
        external_xc = PeriodicExternalXC.prepare(
            basis,
            system,
            func,
            grid_options=external_xc_grid_options,
            image_radius_bohr=external_xc_image_radius_bohr,
            lattice_options=external_lattice_options,
        )

    # DIIS state.
    diis = _make_diis(use_diis, diis_subspace_size)

    # DFT+U setup.
    from .dft_plus_u import (
        _v_ao_per_spin as _dftu_v_ao_per_spin,
    )
    from .dft_plus_u import (
        ao_group_indices as _dftu_ao_group_indices,
    )
    from .dft_plus_u import (
        compute_dudarev_energy as _dftu_compute_dudarev_energy,
    )
    from .dft_plus_u import (
        compute_occupation_matrices as _dftu_compute_occupation_matrices,
    )

    _dftu_sites = list(dft_plus_u_sites or ())
    _dftu_ao_groups = {}
    if _dftu_sites:
        _dftu_ao_groups = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (_site.atom_index, _site.l)
            if _key not in _dftu_ao_groups:
                raise ValueError(
                    f"run_periodic_rhf_gapw: HubbardSite{_key} has no AOs in the basis."
                )

    # SCF iteration.
    E_prev = 0.0
    converged = False
    n_iter = 0
    F = None
    C = np.zeros((basis.nbasis, basis.nbasis))
    e = np.zeros(basis.nbasis)
    occ_final = np.zeros(basis.nbasis)
    occ_final[:n_occ] = 2.0
    fermi_level = 0.0
    entropy = 0.0
    scf_trace: list[dict] = []

    for it in range(1, max_iter + 1):
        # GAPW J build (smooth FFT + augmentation correction).
        J = gapw_builder.build_J(D)

        # Exchange.
        if ex_frac > 0.0:
            lo = _core.LatticeSumOptions()
            lo.cutoff_bohr = 25.0
            jk_mol = _core.build_jk_gamma_molecular_limit(
                basis,
                system,
                lo,
                D,
            )
            K = np.asarray(jk_mol.K)
        else:
            K = np.zeros_like(D)

        F = Hcore + J - 0.5 * ex_frac * K

        # DFT+U Fock contribution.
        e_dft_plus_u = 0.0
        if _dftu_sites:
            P_sigma = 0.5 * D
            n_sigma_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                P_sigma,
                S,
                _dftu_ao_groups,
            )
            e_sigma = _dftu_compute_dudarev_energy(
                _dftu_sites,
                n_sigma_map,
            )
            e_dft_plus_u = 2.0 * float(e_sigma)
            V_AO_sigma = _dftu_v_ao_per_spin(
                _dftu_sites,
                P_sigma,
                S,
                _dftu_ao_groups,
            )
            V_U_fock = S @ V_AO_sigma @ S
            F = F + V_U_fock

        # XC piece. All grid primitives reuse the builder's per-SCF chi cache.
        if func is not None:
            if external_xc is not None:
                # Pöschel et al. 2026, arXiv:2608.19033, Sec. II B-C:
                # evaluate the complete AO density once.  The GAPW
                # smooth/hard/soft split remains a Hartree construction and
                # must not be applied separately to a non-local XC model.
                e_xc_total_iter, V_xc = external_xc.build_gamma(D)
            else:
                _cc = gapw_builder._collocation_cache
                # GAPW: generate the smooth-grid XC field from the SOFT density
                # (mirrors build_J) so the per-atom XC augmentation telescopes
                # to the all-electron XC; project onto the FULL AO basis below.
                _gb, _Dg, _gcc = gapw_builder._smooth_xc_generation(D)
                rho_grid = collocate_density_on_grid(_gb, _Dg, grid, cache=_gcc)
                e_xc_iter, v_xc_grid, v_sigma_grid, grad_rho, v_tau_grid = (
                    _evaluate_xc_on_grid(
                        rho_grid,
                        grid,
                        func,
                        basis=_gb,
                        density_matrix=_Dg,
                        cache=_gcc,
                    )
                )
                if one_centre == "analytic" and _gb is not basis:
                    # Exact derivative of E_xc_sm[rho_tilde(D_soft)]: the
                    # potential couples to soft AO pairs ONLY, embedded in
                    # the soft block.
                    V_soft_blk = _project_vxc_to_ao(
                        _gb,
                        v_xc_grid,
                        grid,
                        v_sigma_grid=v_sigma_grid,
                        grad_rho=grad_rho,
                        cache=_gcc,
                    )
                    V_xc = np.zeros((basis.nbasis, basis.nbasis))
                    _sidx = np.asarray(
                        gapw_builder._aug._soft_indices, dtype=int
                    )
                    V_xc[np.ix_(_sidx, _sidx)] = V_soft_blk
                else:
                    V_xc = _project_vxc_to_ao(
                        basis,
                        v_xc_grid,
                        grid,
                        v_sigma_grid=v_sigma_grid,
                        grad_rho=grad_rho,
                        cache=_cc,
                    )
                if v_tau_grid is not None:
                    # t Fock term: 1/2 ∫ v_tau gradchi.gradchi.
                    if _gb is basis:
                        V_tau = _project_vtau_to_ao(
                            basis, v_tau_grid, grid, cache=_cc
                        )
                    else:
                        V_tau = _project_vtau_to_ao(
                            basis, v_tau_grid, grid
                        )
                    V_xc = V_xc + V_tau
                # Add the per-atom semilocal XC augmentation correction.
                if one_centre == "analytic":
                    xc_corrections, e_xc_aug_iter = (
                        gapw_builder._analytic_xc_correction(D, func)
                    )
                else:
                    xc_corrections, e_xc_aug_iter = (
                        gapw_builder._aug.compute_xc_correction(
                            D,
                            func,
                            return_energy=True,
                        )
                    )
                for delta_Vxc in xc_corrections.values():
                    V_xc = V_xc + delta_Vxc
                e_xc_total_iter = float(e_xc_iter) + float(e_xc_aug_iter)
            F = F + V_xc
        else:
            e_xc_iter = 0.0
            e_xc_total_iter = 0.0

        # Energy.
        if func is None:
            E_elec = 0.5 * float(np.einsum("ij,ij->", D, Hcore + F))
            if _dftu_sites:
                E_elec -= 0.5 * float(np.einsum("ij,ij->", D, V_U_fock))
            E = E_elec + E_nn + e_dft_plus_u
        else:
            E_elec = (
                float(np.einsum("ij,ij->", D, Hcore))
                + 0.5 * float(np.einsum("ij,ij->", D, J))
                - 0.25 * ex_frac * float(np.einsum("ij,ij->", D, K))
                + e_xc_total_iter
            )
            E = E_elec + E_nn + e_dft_plus_u

        # DIIS. ``D`` is the density that built ``F``; see
        # tests/test_diis_error_vector_source.py.
        if diis is not None and it >= diis_start_iter:
            err = F @ D @ S - S @ D @ F
            err = S_half_inv @ err @ S_half_inv
            F = diis.extrapolate(F, err)

        # Solve F C = e S C.
        e, C_orth = _eigh_safe(S_half_inv @ F @ S_half_inv)
        C = S_half_inv @ C_orth

        # Occupations / smearing.
        from .smearing import apply_smearing as _apply_smearing

        sm_result = _apply_smearing(
            [e],
            weights=[1.0],
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_final = np.asarray(sm_result.occupations_per_k[0], dtype=float)
        fermi_level = float(sm_result.mu)
        entropy = float(sm_result.entropy)
        if smearing_T > 0.0:
            D_new = (C * occ_final[None, :]) @ C.T
        else:
            D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        if damping > 0.0:
            D_new = (1.0 - damping) * D_new + damping * D

        dE = E - E_prev
        dD = float(np.linalg.norm(D_new - D))
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_total_iter),
            }
        )
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D = D_new
            break
        D = D_new
        E_prev = E

    # Recompute the smooth and augmented XC components at the returned
    # density. The SCF loop already included both the augmented J and the
    # per-atom XC correction in its Fock and convergence energy; this final
    # pass keeps those XC components tied to a non-converged return as well.
    from dataclasses import replace as _dc_replace

    # Compute the energy breakdown at the converged density using the
    # same components the SCF used (augmented J, same T/S/V_ne).
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))
    E_kin = float(np.einsum("ij,ij->", D, T))
    E_ne = float(np.einsum("ij,ij->", D, V_ne))
    # GAPW Hartree energy in the density-functional form (NOT 1/2 tr(D.J);
    # see GapwJBuilder.gapw_hartree_energy). This already includes the
    # per-atom Hartree augmentation, so the _e_h_aug returned below stays
    # diagnostic-only -- adding it again would double-count.
    E_H = gapw_builder.gapw_hartree_energy(D)
    if ex_frac > 0.0:
        lo = _core.LatticeSumOptions()
        lo.cutoff_bohr = 25.0
        jk_mol = _core.build_jk_gamma_molecular_limit(basis, system, lo, D)
        K_final = np.asarray(jk_mol.K)
        E_K = -0.25 * ex_frac * float(np.einsum("ij,ij->", D, K_final))
    else:
        K_final = np.zeros_like(D)
        E_K = 0.0

    # Re-evaluate the smooth XC energy at the returned density. The last SCF
    # iteration evaluated it at the input density that built the Fock; those
    # densities agree at convergence but are not identical on a non-converged
    # return.
    if func is not None:
        if external_xc is not None:
            e_xc_smooth, _V_xc_final = external_xc.build_gamma(D)
        else:
            _gb, _Dg, _gcc = gapw_builder._smooth_xc_generation(D)
            rho_grid = collocate_density_on_grid(_gb, _Dg, grid, cache=_gcc)
            e_xc_smooth, *_ = _evaluate_xc_on_grid(
                rho_grid,
                grid,
                func,
                basis=_gb,
                density_matrix=_Dg,
                cache=_gcc,
            )
            e_xc_smooth = float(e_xc_smooth)
    else:
        e_xc_smooth = 0.0

    # XC augmentation energy (zero for HF; non-zero for KS-DFT).
    # Analytic mode: projector one-centre densities, matching the SCF
    # loop's per-iteration correction.
    if external_xc is not None:
        e_xc_aug = 0.0
    elif one_centre == "analytic" and func is not None:
        _xc_corr_final, e_xc_aug = gapw_builder._analytic_xc_correction(
            D, func)
    else:
        _e_h_aug, e_xc_aug = gapw_builder._aug.compute_augmentation_energy(
            D,
            functional=func,
        )

    # The SCF loop's +U value belongs to the density that built the final
    # trial Fock.  Recompute it from the accepted return density just as the
    # nonlinear Hartree and XC terms above are recomputed.  This matters for
    # finite-iteration returns and prevents external-XC +U totals from mixing
    # two different densities.
    e_dft_plus_u = 0.0
    if _dftu_sites:
        P_sigma_final = 0.5 * D
        n_sigma_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            P_sigma_final,
            S,
            _dftu_ao_groups,
        )
        e_dft_plus_u = 2.0 * float(
            _dftu_compute_dudarev_energy(
                _dftu_sites,
                n_sigma_final,
            )
        )

    e_total = E_kin + E_ne + E_H + E_K + e_xc_smooth + e_xc_aug + E_nn + e_dft_plus_u

    # Build a breakdown from the same components.
    breakdown = GpwEnergyBreakdown(
        e_kinetic=E_kin,
        e_nuclear_attraction=E_ne,
        e_hartree=E_H,
        e_hf_exchange=E_K,
        e_nuclear_repulsion=E_nn,
        e_total=e_total,
        grid=grid,
        e_xc=e_xc_smooth + e_xc_aug,
        functional=functional,
        e_dft_plus_u=e_dft_plus_u,
    )

    # Diagnostic: Hartree augmentation from J_aug - J_gpw.
    gpw_J = gapw_builder._gpw.build_J(D)
    gapw_J_diag = gapw_builder.build_J(D)
    gapw_correction = 0.5 * float(np.einsum("ij,ij->", D, gapw_J_diag - gpw_J))

    if external_xc is not None:
        # Canonicalize the physical GAPW KS operator at the accepted density.
        # The iterative Fock may be DIIS-extrapolated and belong to the prior
        # density generation; it must not be exposed as F[D] in the result.
        F_final = Hcore + gapw_J_diag - 0.5 * ex_frac * K_final
        if _dftu_sites:
            V_AO_sigma_final = _dftu_v_ao_per_spin(
                _dftu_sites,
                0.5 * D,
                S,
                _dftu_ao_groups,
            )
            F_final = F_final + S @ V_AO_sigma_final @ S
        F_final = F_final + _V_xc_final
        F_final = 0.5 * (F_final + F_final.T)
        e, C_orth = _eigh_safe(S_half_inv @ F_final @ S_half_inv)
        C = S_half_inv @ C_orth
        from .smearing import apply_smearing as _apply_smearing

        final_smearing = _apply_smearing(
            [e],
            weights=[1.0],
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_final = np.asarray(
            final_smearing.occupations_per_k[0], dtype=float
        )
        fermi_level = float(final_smearing.mu)
        entropy = float(final_smearing.entropy)
        F = F_final

    # Dispersion.
    from .periodic_gapw_j import _compute_d3_correction

    e_disp = _compute_d3_correction(system, dispersion, dispersion_functional)
    if e_disp != 0.0:
        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_disp,
            e_dispersion=e_disp,
        )
        e_total = breakdown.e_total

    return GapwScfResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=e_total,
        breakdown=breakdown,
        density=D,
        mo_coeffs=C,
        mo_energies=e,
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        gapw_correction=gapw_correction,
        one_centre=one_centre,
        e_dft_plus_u=e_dft_plus_u,
        dft_plus_u_sites=tuple(dft_plus_u_sites or ()),
        scf_trace=tuple(scf_trace),
        smearing_temperature=smearing_T,
        smearing_entropy=entropy,
        fermi_level=fermi_level,
        occupations=tuple(occ_final.tolist()),
        fock=F,
        overlap=S,
        molecular_limit_declared=molecular_limit_declared,
    )


def run_periodic_rks_gapw(system, basis, *, functional, **kwargs):
    """Closed-shell Γ-only RKS-DFT SCF via the GAPW all-electron route.

    Thin, explicit alias for :func:`run_periodic_rhf_gapw` with a required
    ``functional=``. All other keyword arguments are forwarded unchanged;
    returns the same :class:`GapwScfResult`.  The inherited
    ``one_centre="auto"`` policy resolves to ``"block"`` for RKS; analytic
    DFT remains an explicit, single-atom-only diagnostic.

    Example
    -------
    >>> result = run_periodic_rks_gapw(system, basis, functional="lda")
    """
    if not functional:
        raise ValueError(
            "run_periodic_rks_gapw requires a functional= (e.g. 'lda', "
            "'pbe', 'b3lyp'); for Hartree-Fock use run_periodic_rhf_gapw()."
        )
    return run_periodic_rhf_gapw(system, basis, functional=functional, **kwargs)


# ============================================================
# Open-shell (UHF / UKS) GAPW result dataclasses and entry points
# ============================================================


@dataclass(frozen=True)
class GapwUhfScfResult:
    """Result of a converged single-point GAPW periodic UHF SCF.

    Mirrors :class:`GpwUhfScfResult` but with all-electron GAPW energies.
    ``breakdown.e_total`` includes the augmentation correction.

    Attributes
    ----------
    energy
        Total periodic UHF GAPW energy in Hartree.
    breakdown
        :class:`GpwEnergyBreakdown` at the converged spin densities
        (includes the GAPW correction).
    density_alpha, density_beta
        Per-spin converged AO density matrices.
    mo_coeffs_alpha, mo_coeffs_beta
        MO coefficient matrices per spin.
    mo_energies_alpha, mo_energies_beta
        MO eigenvalues per spin.
    n_alpha, n_beta
        Number of occupied a / b MOs.
    converged
        True iff both energy and density criteria were satisfied.
    n_iter
        Number of SCF iterations.
    grid
        FFT grid used.
    gapw_correction
        The augmentation correction energy (Hartree).
    one_centre
        Resolved one-centre construction used by the SCF.
    fock_alpha, fock_beta, overlap
        Post-DIIS spin Focks and Gamma overlap used for the returned
        eigenpairs.
    molecular_limit_declared
        Whether the molecular-limit analytic envelope was declared.
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    n_alpha: int
    n_beta: int
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    gapw_correction: float = 0.0
    e_dft_plus_u: float = 0.0
    dft_plus_u_sites: tuple = ()
    scf_trace: tuple = ()
    fock_alpha: Optional[np.ndarray] = None
    fock_beta: Optional[np.ndarray] = None
    overlap: Optional[np.ndarray] = None
    one_centre: str = "block"
    molecular_limit_declared: bool = False

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


@dataclass(frozen=True)
class GapwUksScfResult:
    """Result of a converged single-point GAPW periodic UKS SCF.

    Same shape as :class:`GapwUhfScfResult`; ``breakdown.e_xc`` is
    non-zero and ``breakdown.functional`` carries the libxc functional.
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    n_alpha: int
    n_beta: int
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    gapw_correction: float = 0.0
    e_dft_plus_u: float = 0.0
    dft_plus_u_sites: tuple = ()
    scf_trace: tuple = ()
    fock_alpha: Optional[np.ndarray] = None
    fock_beta: Optional[np.ndarray] = None
    overlap: Optional[np.ndarray] = None
    one_centre: str = "block"
    molecular_limit_declared: bool = False

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def run_periodic_uhf_gapw(
    system,
    basis,
    *,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    dft_plus_u_sites=None,
    lmax: int = _DEFAULT_LMAX,
    soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
    n_radial: int = _DEFAULT_N_RADIAL,
    lebedev_order: int = _DEFAULT_LEBEDEV_ORDER,
    one_centre: str = "auto",
    molecular_limit: bool = False,
    memory_override: bool = False,
    quiet: bool = False,
) -> GapwUhfScfResult:
    """Open-shell periodic UHF on the GAPW all-electron route (Γ-only).

    Pure Hartree-Fock with per-atom Hartree augmentation via
    :class:`GapwJBuilder`. Follows the same open-shell conventions as
    :func:`vibeqc.periodic_gapw_open_shell.run_periodic_uhf_gpw` but
    uses the GAPW J build for all-electron accuracy.

    ``one_centre="auto"`` (default) selects the fit-free analytic-ERI
    Hartree augmentation when ``molecular_limit=True``, exactly as on the HF
    branch of :func:`run_periodic_rhf_gapw`. The Hartree only ever sees the
    spin-summed density, so the closed-shell validation carries over
    unchanged (the exact Fock keeps ``½ tr(D_total·J) = E_H``).
    Molecular-limit Gamma envelope. ``one_centre="projector"``
    (experimental) is single-atom only, as on the RHF driver.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis.
    n_alpha, n_beta
        Per-spin occupations. Default ``None`` infers from ``system``
        spin multiplicity (requires ``system.multiplicity``).
    grid, cutoff_ha
        FFT grid or cutoff.
    max_iter, conv_tol_energy, conv_tol_density
        SCF convergence parameters.
    damping
        Linear density mixing damping.
    initial_density
        Optional ``(D_alpha_init, D_beta_init)``.
    v_ne_convention, smearing_alpha
        V_ne convention. See :func:`run_periodic_rhf_gapw`.
    use_diis, diis_subspace_size, diis_start_iter
        DIIS knobs.
    lmax, soft_cutoff, n_radial, lebedev_order
        GAPW augmentation grid parameters.
    one_centre
        With ``molecular_limit=True``, ``"auto"`` (default) selects the
        fit-free ``"analytic"`` Hartree construction. Without that declaration
        it fails closed. ``"block"`` retains the historical AO-block mode.
    molecular_limit
        Declare an isolated single-Gamma molecule/atom cell for automatic HF
        analytic selection. Explicit ``"analytic"`` is the expert opt-in
        spelling of the same physical envelope.
    memory_override
        Permit the analytic hard/soft ERI cache to exceed the live-memory
        preflight.
    quiet
        Suppress experimental warnings.

    Returns
    -------
    result
        :class:`GapwUhfScfResult` with all-electron energies.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    input_restart_supplied = initial_density is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver="run_periodic_uhf_gapw",
        supported=periodic_guess_capabilities('gapw', 'UHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
    )
    if initial_density is not None:
        from .guess_read import _real_density
        if len(initial_density) != 2:
            raise ValueError("READ: expected an alpha/beta density pair")
        initial_density = tuple(_real_density(d) for d in initial_density)


    # Experimental-gate warning retired (correctness bugs fixed 2026-06-26).
    if system.dim != 3:
        raise ValueError(f"run_periodic_uhf_gapw: dim == 3 only; got dim={system.dim}")
    if not isinstance(molecular_limit, (bool, np.bool_)):
        raise TypeError(
            "run_periodic_uhf_gapw: molecular_limit must be bool; "
            f"got {type(molecular_limit).__name__}."
        )

    requested_one_centre = str(one_centre).strip().lower()
    one_centre = _resolve_one_centre_mode(
        one_centre,
        functional=None,
        molecular_limit=bool(molecular_limit),
    )
    molecular_limit_declared = bool(
        molecular_limit or requested_one_centre == "analytic"
    )

    from .periodic_gapw_open_shell import infer_alpha_beta_from_system

    n_alpha_resolved, n_beta_resolved = infer_alpha_beta_from_system(
        system,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    # Build grid if needed.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    if one_centre == "analytic":
        _preflight_analytic_eri_memory(
            system,
            basis,
            grid,
            functional=None,
            soft_cutoff=soft_cutoff,
            lmax=lmax,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
            open_shell=True,
            memory_override=memory_override,
        )

    # One-electron integrals.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(
        basis,
        system,
        v_ne_convention,
        smearing_alpha,
        grid,
    )
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne

    # Ewald nuclear repulsion.
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    # Same single-atom envelope as the RHF driver: multi-atom off-centre
    # fits are variationally exploitable (see run_periodic_rhf_gapw).
    if one_centre == "projector" and len(list(system.unit_cell)) > 1:
        raise NotImplementedError(
            "run_periodic_uhf_gapw: one_centre='projector' supports "
            "single-atom cells only for now (off-centre fit errors are "
            "variationally exploitable; see run_periodic_rhf_gapw). "
            "Use one_centre='analytic' or explicit one_centre='block'."
        )

    from .dft_plus_u import (
        _v_ao_per_spin as _dftu_v_ao_per_spin,
        ao_group_indices as _dftu_ao_group_indices,
        compute_dudarev_energy as _dftu_compute_dudarev_energy,
        compute_occupation_matrices as _dftu_compute_occupation_matrices,
    )

    _dftu_sites = tuple(dft_plus_u_sites or ())
    if _dftu_sites:
        _dftu_ao_groups = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (int(_site.atom_index), int(_site.l))
            if _key not in _dftu_ao_groups:
                raise ValueError(
                    f"run_periodic_uhf_gapw: HubbardSite{_key} has no AOs "
                    f"in the basis. Available channels: "
                    f"{sorted(_dftu_ao_groups.keys())}"
                )
    else:
        _dftu_ao_groups = {}

    # GAPW J builder.
    gapw_builder = GapwJBuilder(
        basis,
        system,
        grid,
        lmax=lmax,
        soft_cutoff=soft_cutoff,
        n_radial=n_radial,
        lebedev_order=lebedev_order,
        one_centre=one_centre,
        quiet=quiet,
    )

    # Lattice sum options for per-spin K.
    from .periodic_gapw_open_shell import _k_per_spin

    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0

    # Pre-build orthonormaliser.
    s_eigs, U = _eigh_safe(S)
    S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T

    # Density-mode guesses use the shared periodic adapter; explicit restart
    # densities retain precedence and HCORE keeps the route-local fallback.
    if initial_density is None:
        from .guess import initial_densities_open_shell

        initial_density = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha_resolved,
            n_beta_resolved,
            (_core.InitialGuess.SAD if initial_guess == _core.InitialGuess.PATOM
             else initial_guess),
            is_periodic=True,
            periodic_system=system,
            lattice_opts=_core.LatticeSumOptions(),
            overlap=S,
                              atomic_spins=atomic_spins,
        )
    if initial_density is not None:
        D_alpha_init, D_beta_init = initial_density
        D_alpha = np.asarray(D_alpha_init, dtype=float).copy()
        D_beta = np.asarray(D_beta_init, dtype=float).copy()
        if D_alpha.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density[alpha] shape {D_alpha.shape} "
                f"doesn't match basis ({basis.nbasis}, {basis.nbasis})"
            )
        if D_beta.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density[beta] shape {D_beta.shape} "
                f"doesn't match basis ({basis.nbasis}, {basis.nbasis})"
            )
    else:
        e_init, C_orth_init = _eigh_safe(S_half_inv @ Hcore @ S_half_inv)
        C_init = S_half_inv @ C_orth_init
        D_alpha = C_init[:, :n_alpha_resolved] @ C_init[:, :n_alpha_resolved].T
        D_beta = C_init[:, :n_beta_resolved] @ C_init[:, :n_beta_resolved].T

    from .guess import normalize_spin_density_guess
    D_alpha, D_beta = normalize_spin_density_guess(
        D_alpha, D_beta, S, n_alpha_resolved, n_beta_resolved,
    )

    if initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_spin_density_step
        D_alpha, D_beta = patom_spin_density_step(
            D_alpha, D_beta, Hcore, S, n_alpha_resolved, n_beta_resolved,
            gapw_builder.build_J, lambda d: _k_per_spin(basis, system, d, lo),
        )

    # DIIS state (one joint history couples the spins).
    diis = _make_diis(use_diis, diis_subspace_size)

    n_basis = basis.nbasis
    C_alpha = np.zeros((n_basis, n_basis))
    C_beta = np.zeros((n_basis, n_basis))
    e_a = np.zeros(n_basis)
    e_b = np.zeros(n_basis)

    # SCF iteration.
    E_prev = 0.0
    converged = False
    n_iter = 0
    scf_trace: list[dict] = []

    for it in range(1, max_iter + 1):
        D_total = D_alpha + D_beta

        # GAPW J build (smooth FFT + augmentation correction).
        J = gapw_builder.build_J(D_total)

        # Per-spin exchange.
        K_alpha = _k_per_spin(basis, system, D_alpha, lo)
        K_beta = _k_per_spin(basis, system, D_beta, lo)

        F_alpha = Hcore + J - K_alpha
        F_beta = Hcore + J - K_beta

        e_dft_plus_u_iter = 0.0
        if _dftu_sites:
            n_alpha_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                D_alpha,
                S,
                _dftu_ao_groups,
            )
            n_beta_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                D_beta,
                S,
                _dftu_ao_groups,
            )
            e_dft_plus_u_iter = float(
                _dftu_compute_dudarev_energy(_dftu_sites, n_alpha_map)
                + _dftu_compute_dudarev_energy(_dftu_sites, n_beta_map)
            )
            F_alpha = F_alpha + S @ _dftu_v_ao_per_spin(
                _dftu_sites,
                D_alpha,
                S,
                _dftu_ao_groups,
            ) @ S
            F_beta = F_beta + S @ _dftu_v_ao_per_spin(
                _dftu_sites,
                D_beta,
                S,
                _dftu_ao_groups,
            ) @ S

        # Energy (open-shell convention).
        E_core = float(np.einsum("ij,ij->", D_total, Hcore))
        E_J = 0.5 * float(np.einsum("ij,ij->", D_total, J))
        E_K_a = float(np.einsum("ij,ij->", D_alpha, K_alpha))
        E_K_b = float(np.einsum("ij,ij->", D_beta, K_beta))
        E_K = -0.5 * (E_K_a + E_K_b)
        E_elec = E_core + E_J + E_K
        E = E_elec + E_nn + e_dft_plus_u_iter

        # DIIS per spin. Stacking α and β makes the kernel accumulate one
        # B-matrix from Tr(e_α_iᵀ e_α_j) + Tr(e_β_iᵀ e_β_j), so a single
        # coefficient set drives both Fock matrices.
        if diis is not None and it >= diis_start_iter:
            err_a = F_alpha @ D_alpha @ S - S @ D_alpha @ F_alpha
            err_a = S_half_inv @ err_a @ S_half_inv
            err_b = F_beta @ D_beta @ S - S @ D_beta @ F_beta
            err_b = S_half_inv @ err_b @ S_half_inv
            F_alpha, F_beta = diis.extrapolate_spin_coupled(
                F_alpha, F_beta, err_a, err_b
            )

        # Solve F_s C_s = e_s S C_s.
        e_a, C_orth_a = _eigh_safe(S_half_inv @ F_alpha @ S_half_inv)
        C_alpha = S_half_inv @ C_orth_a
        e_b, C_orth_b = _eigh_safe(S_half_inv @ F_beta @ S_half_inv)
        C_beta = S_half_inv @ C_orth_b

        if n_alpha_resolved > 0:
            D_alpha_new = (
                C_alpha[:, :n_alpha_resolved] @ C_alpha[:, :n_alpha_resolved].T
            )
        else:
            D_alpha_new = np.zeros_like(D_alpha)
        if n_beta_resolved > 0:
            D_beta_new = C_beta[:, :n_beta_resolved] @ C_beta[:, :n_beta_resolved].T
        else:
            D_beta_new = np.zeros_like(D_beta)

        if damping > 0.0:
            D_alpha_new = (1.0 - damping) * D_alpha_new + damping * D_alpha
            D_beta_new = (1.0 - damping) * D_beta_new + damping * D_beta

        dE = E - E_prev
        dD = float(
            np.sqrt(
                np.linalg.norm(D_alpha_new - D_alpha) ** 2
                + np.linalg.norm(D_beta_new - D_beta) ** 2
            )
        )
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": 0.0,
                "e_dft_plus_u": float(e_dft_plus_u_iter),
            }
        )
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D_alpha = D_alpha_new
            D_beta = D_beta_new
            break
        D_alpha = D_alpha_new
        D_beta = D_beta_new
        E_prev = E

    # Final breakdown.
    D_total = D_alpha + D_beta
    J = gapw_builder.build_J(D_total)
    K_alpha = _k_per_spin(basis, system, D_alpha, lo)
    K_beta = _k_per_spin(basis, system, D_beta, lo)

    E_kin = float(np.einsum("ij,ij->", D_total, T))
    E_ne = float(np.einsum("ij,ij->", D_total, V_ne))
    E_H = 0.5 * float(np.einsum("ij,ij->", D_total, J))
    E_K_final = -0.5 * (
        float(np.einsum("ij,ij->", D_alpha, K_alpha))
        + float(np.einsum("ij,ij->", D_beta, K_beta))
    )
    e_dft_plus_u_final = 0.0
    if _dftu_sites:
        n_alpha_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            D_alpha,
            S,
            _dftu_ao_groups,
        )
        n_beta_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            D_beta,
            S,
            _dftu_ao_groups,
        )
        e_dft_plus_u_final = float(
            _dftu_compute_dudarev_energy(_dftu_sites, n_alpha_final)
            + _dftu_compute_dudarev_energy(_dftu_sites, n_beta_final)
        )

    # GAPW total energy: the SCF already used the augmented J in the
    # Fock, so E_H = 0.5 tr(D_total J) already includes the Hartree
    # augmentation. UHF has no XC. No additional term needed.
    e_total = E_kin + E_ne + E_H + E_K_final + E_nn + e_dft_plus_u_final

    # Build the decomposition directly from the augmented operator.  Calling
    # the smooth-GPW energy helper here would allocate and evaluate a second,
    # unaugmented J only to overwrite its values afterwards.
    breakdown = GpwEnergyBreakdown(
        e_kinetic=E_kin,
        e_nuclear_attraction=E_ne,
        e_hartree=E_H,
        e_hf_exchange=E_K_final,
        e_nuclear_repulsion=E_nn,
        e_total=e_total,
        grid=grid,
        e_xc=0.0,
        functional=None,
        e_dft_plus_u=e_dft_plus_u_final,
    )

    # Diagnostic: Hartree augmentation from J_aug - J_gpw.
    gpw_J = gapw_builder._gpw.build_J(D_total)
    gapw_J_diag = gapw_builder.build_J(D_total)
    gapw_correction = 0.5 * float(np.einsum("ij,ij->", D_total, gapw_J_diag - gpw_J))

    return GapwUhfScfResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=breakdown.e_total,
        breakdown=breakdown,
        density_alpha=D_alpha,
        density_beta=D_beta,
        mo_coeffs_alpha=C_alpha,
        mo_coeffs_beta=C_beta,
        mo_energies_alpha=e_a,
        mo_energies_beta=e_b,
        n_alpha=n_alpha_resolved,
        n_beta=n_beta_resolved,
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        gapw_correction=gapw_correction,
        one_centre=one_centre,
        e_dft_plus_u=e_dft_plus_u_final,
        dft_plus_u_sites=_dftu_sites,
        scf_trace=tuple(scf_trace),
        fock_alpha=F_alpha,
        fock_beta=F_beta,
        overlap=S,
        molecular_limit_declared=molecular_limit_declared,
    )


def run_periodic_uks_gapw(
    system,
    basis,
    *,
    functional: str,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    dft_plus_u_sites=None,
    lmax: int = _DEFAULT_LMAX,
    soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
    n_radial: int = _DEFAULT_N_RADIAL,
    lebedev_order: int = _DEFAULT_LEBEDEV_ORDER,
    one_centre: str = "auto",
    memory_override: bool = False,
    quiet: bool = False,
) -> GapwUksScfResult:
    """Open-shell periodic UKS on the GAPW all-electron route (Γ-only).

    Spin-polarised KS-DFT with per-atom Hartree and XC augmentation.
    The smooth-grid XC is evaluated on the FFT grid (polarised path),
    then per-atom XC corrections for each spin channel are added via
    :meth:`GapwAugmentation.compute_xc_correction_polarised`.

    ``one_centre="analytic"`` (experimental) uses the fit-free
    analytic-ERI Hartree augmentation plus the spin-polarised
    projector-density XC augmentation
    (:func:`vibeqc.periodic_gapw_projector.projector_xc_correction_polarised`),
    exactly mirroring the closed-shell analytic DFT path. The same
    envelope applies: single-atom cells only (fitted off-centre XC
    densities are variationally soft, and the softness grows with
    basis richness; see :func:`run_periodic_rhf_gapw`).
    ``one_centre="projector"`` is HF-only and rejected here.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis.
    functional
        libxc functional name (required, e.g. ``'lda'``, ``'pbe'``,
        ``'b3lyp'``).
    n_alpha, n_beta
        Per-spin occupations.
    grid, cutoff_ha
        FFT grid or cutoff.
    max_iter, conv_tol_energy, conv_tol_density
        SCF convergence parameters.
    damping
        Linear density mixing damping.
    initial_density
        Optional ``(D_alpha_init, D_beta_init)``.
    v_ne_convention, smearing_alpha
        V_ne convention.
    use_diis, diis_subspace_size, diis_start_iter
        DIIS knobs.
    lmax, soft_cutoff, n_radial, lebedev_order
        GAPW augmentation grid parameters.
    one_centre
        ``"auto"`` (default) resolves to ``"block"`` and retains the
        validated UKS construction.
        ``"analytic"`` is an explicit single-atom-only diagnostic;
        ``"projector"`` is rejected because that mode is HF-only.
    memory_override
        Permit the analytic hard/soft ERI cache to exceed the live-memory
        preflight when explicit ``one_centre="analytic"`` is selected.
    quiet
        Suppress experimental warnings.

    Returns
    -------
    result
        :class:`GapwUksScfResult` with all-electron energies.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    input_restart_supplied = initial_density is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver="run_periodic_uks_gapw",
        supported=periodic_guess_capabilities('gapw', 'UKS', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
    )
    if initial_density is not None:
        from .guess_read import _real_density
        if len(initial_density) != 2:
            raise ValueError("READ: expected an alpha/beta density pair")
        initial_density = tuple(_real_density(d) for d in initial_density)


    if not functional:
        raise ValueError(
            "run_periodic_uks_gapw: functional is required (got "
            f"functional={functional!r}). For pure UHF use "
            "run_periodic_uhf_gapw."
        )

    # Experimental-gate warning retired (correctness bugs fixed 2026-06-26).
    if system.dim != 3:
        raise ValueError(f"run_periodic_uks_gapw: dim == 3 only; got dim={system.dim}")

    one_centre = _resolve_one_centre_mode(
        one_centre,
        functional=str(functional),
        molecular_limit=False,
    )

    from .periodic_gapw_open_shell import (
        _evaluate_xc_polarised_on_grid,
        _k_per_spin,
        _project_vxc_polarised_to_ao,
        infer_alpha_beta_from_system,
    )

    n_alpha_resolved, n_beta_resolved = infer_alpha_beta_from_system(
        system,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    # Build grid if needed.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    if one_centre == "analytic":
        _preflight_analytic_eri_memory(
            system,
            basis,
            grid,
            functional=functional,
            soft_cutoff=soft_cutoff,
            lmax=lmax,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
            open_shell=True,
            memory_override=memory_override,
        )

    # One-electron integrals.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(
        basis,
        system,
        v_ne_convention,
        smearing_alpha,
        grid,
    )
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne

    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    if one_centre == "projector":
        raise NotImplementedError(
            "run_periodic_uks_gapw: one_centre='projector' is HF-only "
            "(and single-atom only); use one_centre='analytic' or the "
            "default 'block'."
        )
    # Same calibration gate as the closed-shell analytic DFT path
    # (2026-08-01): single-atom cells only. Multi-atom XC telescoping
    # runs on fitted off-centre one-centre densities whose variational
    # softness GROWS with basis richness (LiH LDA -10.6 -> -105 mHa
    # from STO-3G to def2-SVP; H2O up to -21 Ha), so no
    # neighbour-count or basis-quality gate can bound that class.
    if one_centre == "analytic" and len(list(system.unit_cell)) > 1:
        raise NotImplementedError(
            "run_periodic_uks_gapw: one_centre='analytic' with "
            f"functional={functional!r} supports single-atom cells "
            "only (same envelope as run_periodic_rhf_gapw with a "
            "functional: fitted off-centre XC densities are "
            "variationally soft, and the softness grows with basis "
            "richness). Use jk_method='gdf'/'bipole' for DFT on "
            "molecules."
        )

    from .dft_plus_u import (
        _v_ao_per_spin as _dftu_v_ao_per_spin,
        ao_group_indices as _dftu_ao_group_indices,
        compute_dudarev_energy as _dftu_compute_dudarev_energy,
        compute_occupation_matrices as _dftu_compute_occupation_matrices,
    )

    _dftu_sites = tuple(dft_plus_u_sites or ())
    if _dftu_sites:
        _dftu_ao_groups = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (int(_site.atom_index), int(_site.l))
            if _key not in _dftu_ao_groups:
                raise ValueError(
                    f"run_periodic_uks_gapw: HubbardSite{_key} has no AOs "
                    f"in the basis. Available channels: "
                    f"{sorted(_dftu_ao_groups.keys())}"
                )
    else:
        _dftu_ao_groups = {}

    # GAPW J builder.
    gapw_builder = GapwJBuilder(
        basis,
        system,
        grid,
        lmax=lmax,
        soft_cutoff=soft_cutoff,
        n_radial=n_radial,
        lebedev_order=lebedev_order,
        one_centre=one_centre,
        quiet=quiet,
    )

    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0

    # Functional (spin=2 for polarised path).
    func = _core.Functional(functional, 2)
    # Full-range-only K on this route (see run_periodic_rhf_gpw).
    reject_unscreened_range_separated(func, where="run_periodic_uks_gapw")
    ex_frac = float(func.hf_exchange_fraction)
    # meta-GGA is enabled now that the open-shell smooth XC telescopes
    # correctly (it generates from the soft density per spin, below, exactly
    # as the closed-shell RKS driver does). The spin-polarised per-atom τ
    # augmentation + smooth per-spin v_tau Fock term are already wired
    # (compute_*_polarised handle meta-GGA). Stiff (non-self-regularising)
    # meta-GGAs still fail closed upstream via the smooth-grid stiffness guard
    # in _evaluate_xc_polarised_on_grid, which runs first.

    # Pre-build orthonormaliser.
    s_eigs, U = _eigh_safe(S)
    S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T

    # Density-mode guesses use the shared periodic adapter; explicit restart
    # densities retain precedence and HCORE keeps the route-local fallback.
    if initial_density is None:
        from .guess import initial_densities_open_shell

        initial_density = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha_resolved,
            n_beta_resolved,
            (_core.InitialGuess.SAD if initial_guess == _core.InitialGuess.PATOM
             else initial_guess),
            is_periodic=True,
            periodic_system=system,
            lattice_opts=_core.LatticeSumOptions(),
            overlap=S,
                              atomic_spins=atomic_spins,
        )
    if initial_density is not None:
        D_alpha_init, D_beta_init = initial_density
        D_alpha = np.asarray(D_alpha_init, dtype=float).copy()
        D_beta = np.asarray(D_beta_init, dtype=float).copy()
    else:
        e_init, C_orth_init = _eigh_safe(S_half_inv @ Hcore @ S_half_inv)
        C_init = S_half_inv @ C_orth_init
        D_alpha = C_init[:, :n_alpha_resolved] @ C_init[:, :n_alpha_resolved].T
        D_beta = C_init[:, :n_beta_resolved] @ C_init[:, :n_beta_resolved].T

    from .guess import normalize_spin_density_guess
    D_alpha, D_beta = normalize_spin_density_guess(
        D_alpha, D_beta, S, n_alpha_resolved, n_beta_resolved,
    )

    if initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_spin_density_step
        D_alpha, D_beta = patom_spin_density_step(
            D_alpha, D_beta, Hcore, S, n_alpha_resolved, n_beta_resolved,
            gapw_builder.build_J, lambda d: _k_per_spin(basis, system, d, lo),
        )

    # DIIS state (one joint history couples the spins).
    diis = _make_diis(use_diis, diis_subspace_size)

    n_basis = basis.nbasis
    C_alpha = np.zeros((n_basis, n_basis))
    C_beta = np.zeros((n_basis, n_basis))
    e_a = np.zeros(n_basis)
    e_b = np.zeros(n_basis)

    # SCF iteration.
    E_prev = 0.0
    converged = False
    n_iter = 0
    scf_trace: list[dict] = []

    for it in range(1, max_iter + 1):
        D_total = D_alpha + D_beta

        # GAPW J build.
        J = gapw_builder.build_J(D_total)

        # Per-spin exchange.
        if ex_frac > 0.0:
            K_alpha = _k_per_spin(basis, system, D_alpha, lo)
            K_beta = _k_per_spin(basis, system, D_beta, lo)
        else:
            K_alpha = np.zeros_like(D_alpha)
            K_beta = np.zeros_like(D_beta)

        F_alpha = Hcore + J - ex_frac * K_alpha
        F_beta = Hcore + J - ex_frac * K_beta

        e_dft_plus_u_iter = 0.0
        if _dftu_sites:
            n_alpha_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                D_alpha,
                S,
                _dftu_ao_groups,
            )
            n_beta_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                D_beta,
                S,
                _dftu_ao_groups,
            )
            e_dft_plus_u_iter = float(
                _dftu_compute_dudarev_energy(_dftu_sites, n_alpha_map)
                + _dftu_compute_dudarev_energy(_dftu_sites, n_beta_map)
            )
            F_alpha = F_alpha + S @ _dftu_v_ao_per_spin(
                _dftu_sites,
                D_alpha,
                S,
                _dftu_ao_groups,
            ) @ S
            F_beta = F_beta + S @ _dftu_v_ao_per_spin(
                _dftu_sites,
                D_beta,
                S,
                _dftu_ao_groups,
            ) @ S

        # XC on the smooth FFT grid (polarised path). GAPW: generate the
        # smooth-grid XC field from the SOFT (pseudo) density per spin --
        # exactly as the closed-shell RKS driver does via
        # _smooth_xc_generation -- so the per-atom XC augmentation
        # E_xc[ρ_a] − E_xc[ρ̃_a] telescopes to the all-electron XC. Collocating
        # the full (hard) density here computes smooth(hard) + (hard−soft) =
        # 2·hard − soft, double-counting the hard core (~0.3 Ha over-binding on
        # He/STO-3G, LDA/PBE/meta-GGA alike). v_tau below is likewise generated
        # from the soft density; both V_xc and V_tau still project onto the
        # FULL AO basis (soft only restricts what GENERATES the field).
        _cc = gapw_builder._collocation_cache
        _gb, _Da_g, _gcc = gapw_builder._smooth_xc_generation(D_alpha)
        _, _Db_g, _ = gapw_builder._smooth_xc_generation(D_beta)
        rho_a_grid = collocate_density_on_grid(_gb, _Da_g, grid, cache=_gcc)
        rho_b_grid = collocate_density_on_grid(_gb, _Db_g, grid, cache=_gcc)
        (
            e_xc_iter,
            v_xc_a_grid,
            v_xc_b_grid,
            v_sigma_aa_g,
            v_sigma_ab_g,
            v_sigma_bb_g,
            grad_a,
            grad_b,
            _v_tau_a_aug,
            _v_tau_b_aug,
        ) = _evaluate_xc_polarised_on_grid(
            rho_a_grid,
            rho_b_grid,
            grid,
            func,
            basis=_gb,
            density_matrix_alpha=_Da_g,
            density_matrix_beta=_Db_g,
            cache=_gcc,
        )
        if one_centre == "analytic" and _gb is not basis:
            # Exact derivative of E_xc_sm[rho_tilde(D_soft)] per spin:
            # the potential couples to soft AO pairs ONLY, embedded in
            # the soft block. Projecting onto the full basis adds
            # hard-AO elements that are NOT dE/dD of this energy (the
            # closed-shell driver's Ne/6-31G walked +373 mHa uphill
            # before the same fix).
            _sidx = np.asarray(gapw_builder._aug._soft_indices, dtype=int)
            V_soft_a = _project_vxc_polarised_to_ao(
                _gb,
                "alpha",
                v_xc_a_grid,
                grid,
                v_sigma_self=v_sigma_aa_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_a,
                grad_rho_other=grad_b,
                cache=_gcc,
            )
            V_soft_b = _project_vxc_polarised_to_ao(
                _gb,
                "beta",
                v_xc_b_grid,
                grid,
                v_sigma_self=v_sigma_bb_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_b,
                grad_rho_other=grad_a,
                cache=_gcc,
            )
            V_xc_alpha = np.zeros((basis.nbasis, basis.nbasis))
            V_xc_beta = np.zeros((basis.nbasis, basis.nbasis))
            V_xc_alpha[np.ix_(_sidx, _sidx)] = V_soft_a
            V_xc_beta[np.ix_(_sidx, _sidx)] = V_soft_b
        else:
            V_xc_alpha = _project_vxc_polarised_to_ao(
                basis,
                "alpha",
                v_xc_a_grid,
                grid,
                v_sigma_self=v_sigma_aa_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_a,
                grad_rho_other=grad_b,
                cache=_cc,
            )
            V_xc_beta = _project_vxc_polarised_to_ao(
                basis,
                "beta",
                v_xc_b_grid,
                grid,
                v_sigma_self=v_sigma_bb_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_b,
                grad_rho_other=grad_a,
                cache=_cc,
            )
        # Meta-GGA: smooth-grid per-spin t Fock term (the per-atom t
        # augmentation is added by compute_xc_correction_polarised below).
        if _v_tau_a_aug is not None:
            from .periodic_gapw_j import _project_vtau_to_ao

            V_xc_alpha = V_xc_alpha + _project_vtau_to_ao(
                basis, _v_tau_a_aug, grid, cache=_cc
            )
            V_xc_beta = V_xc_beta + _project_vtau_to_ao(
                basis, _v_tau_b_aug, grid, cache=_cc
            )

        # Add per-atom XC augmentation correction for each spin. In
        # analytic mode the one-centre densities come from the projector
        # expansions (block densities mis-telescope XC by Ha-scale on
        # bond-inside-sphere systems; see the closed-shell driver).
        if one_centre == "analytic":
            (
                xc_corr_alpha,
                xc_corr_beta,
                e_xc_aug_iter,
            ) = gapw_builder._analytic_xc_correction_polarised(
                D_alpha,
                D_beta,
                func,
            )
        else:
            (
                xc_corr_alpha,
                xc_corr_beta,
                e_xc_aug_iter,
            ) = gapw_builder._aug.compute_xc_correction_polarised(
                D_alpha,
                D_beta,
                func,
                return_energy=True,
            )
        for delta_V in xc_corr_alpha.values():
            V_xc_alpha = V_xc_alpha + delta_V
        for delta_V in xc_corr_beta.values():
            V_xc_beta = V_xc_beta + delta_V

        F_alpha = F_alpha + V_xc_alpha
        F_beta = F_beta + V_xc_beta
        e_xc_total_iter = float(e_xc_iter) + float(e_xc_aug_iter)

        # Energy (open-shell convention).
        E_core = float(np.einsum("ij,ij->", D_total, Hcore))
        E_J = 0.5 * float(np.einsum("ij,ij->", D_total, J))
        E_K_a = float(np.einsum("ij,ij->", D_alpha, K_alpha))
        E_K_b = float(np.einsum("ij,ij->", D_beta, K_beta))
        E_K = -0.5 * ex_frac * (E_K_a + E_K_b)
        E_elec = E_core + E_J + E_K + e_xc_total_iter
        E = E_elec + E_nn + e_dft_plus_u_iter

        # DIIS per spin. Stacking α and β makes the kernel accumulate one
        # B-matrix from Tr(e_α_iᵀ e_α_j) + Tr(e_β_iᵀ e_β_j), so a single
        # coefficient set drives both Fock matrices.
        if diis is not None and it >= diis_start_iter:
            err_a = F_alpha @ D_alpha @ S - S @ D_alpha @ F_alpha
            err_a = S_half_inv @ err_a @ S_half_inv
            err_b = F_beta @ D_beta @ S - S @ D_beta @ F_beta
            err_b = S_half_inv @ err_b @ S_half_inv
            F_alpha, F_beta = diis.extrapolate_spin_coupled(
                F_alpha, F_beta, err_a, err_b
            )

        # Solve F_s C_s = e_s S C_s.
        e_a, C_orth_a = _eigh_safe(S_half_inv @ F_alpha @ S_half_inv)
        C_alpha = S_half_inv @ C_orth_a
        e_b, C_orth_b = _eigh_safe(S_half_inv @ F_beta @ S_half_inv)
        C_beta = S_half_inv @ C_orth_b

        if n_alpha_resolved > 0:
            D_alpha_new = (
                C_alpha[:, :n_alpha_resolved] @ C_alpha[:, :n_alpha_resolved].T
            )
        else:
            D_alpha_new = np.zeros_like(D_alpha)
        if n_beta_resolved > 0:
            D_beta_new = C_beta[:, :n_beta_resolved] @ C_beta[:, :n_beta_resolved].T
        else:
            D_beta_new = np.zeros_like(D_beta)

        if damping > 0.0:
            D_alpha_new = (1.0 - damping) * D_alpha_new + damping * D_alpha
            D_beta_new = (1.0 - damping) * D_beta_new + damping * D_beta

        dE = E - E_prev
        dD = float(
            np.sqrt(
                np.linalg.norm(D_alpha_new - D_alpha) ** 2
                + np.linalg.norm(D_beta_new - D_beta) ** 2
            )
        )
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_total_iter),
                "e_dft_plus_u": float(e_dft_plus_u_iter),
            }
        )
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D_alpha = D_alpha_new
            D_beta = D_beta_new
            break
        D_alpha = D_alpha_new
        D_beta = D_beta_new
        E_prev = E

    # Final breakdown at converged densities.
    D_total = D_alpha + D_beta
    J = gapw_builder.build_J(D_total)
    if ex_frac > 0.0:
        K_alpha = _k_per_spin(basis, system, D_alpha, lo)
        K_beta = _k_per_spin(basis, system, D_beta, lo)
    else:
        K_alpha = np.zeros_like(D_alpha)
        K_beta = np.zeros_like(D_beta)

    # Final smooth-grid XC energy from the SOFT density per spin (same GAPW
    # telescoping as the SCF loop above / the RKS driver): the per-atom
    # augmentation added below via compute_augmentation_energy_polarised only
    # telescopes to the all-electron XC when the smooth term is E_xc[soft].
    _cc = gapw_builder._collocation_cache
    _gb, _Da_g, _gcc = gapw_builder._smooth_xc_generation(D_alpha)
    _, _Db_g, _ = gapw_builder._smooth_xc_generation(D_beta)
    rho_a_grid = collocate_density_on_grid(_gb, _Da_g, grid, cache=_gcc)
    rho_b_grid = collocate_density_on_grid(_gb, _Db_g, grid, cache=_gcc)
    (e_xc_final, *_rest) = _evaluate_xc_polarised_on_grid(
        rho_a_grid,
        rho_b_grid,
        grid,
        func,
        basis=_gb,
        density_matrix_alpha=_Da_g,
        density_matrix_beta=_Db_g,
        cache=_gcc,
    )

    E_kin = float(np.einsum("ij,ij->", D_total, T))
    E_ne = float(np.einsum("ij,ij->", D_total, V_ne))
    E_H = 0.5 * float(np.einsum("ij,ij->", D_total, J))
    E_K_final = (
        -0.5
        * ex_frac
        * (
            float(np.einsum("ij,ij->", D_alpha, K_alpha))
            + float(np.einsum("ij,ij->", D_beta, K_beta))
        )
    )
    e_dft_plus_u_final = 0.0
    if _dftu_sites:
        n_alpha_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            D_alpha,
            S,
            _dftu_ao_groups,
        )
        n_beta_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            D_beta,
            S,
            _dftu_ao_groups,
        )
        e_dft_plus_u_final = float(
            _dftu_compute_dudarev_energy(_dftu_sites, n_alpha_final)
            + _dftu_compute_dudarev_energy(_dftu_sites, n_beta_final)
        )

    # Recompute the augmented XC energy at the returned spin densities. The
    # SCF loop already included it in its convergence energy; this final pass
    # keeps the returned breakdown tied to the returned densities.
    if one_centre == "analytic":
        _corr_a_final, _corr_b_final, e_xc_aug = (
            gapw_builder._analytic_xc_correction_polarised(
                D_alpha,
                D_beta,
                func,
            )
        )
    else:
        _e_h_aug, e_xc_aug = (
            gapw_builder._aug.compute_augmentation_energy_polarised(
                D_alpha,
                D_beta,
                func,
            )
        )
    e_total = (
        E_kin + E_ne + E_H + E_K_final + float(e_xc_final)
        + E_nn + e_xc_aug + e_dft_plus_u_final
    )

    breakdown = GpwEnergyBreakdown(
        e_kinetic=E_kin,
        e_nuclear_attraction=E_ne,
        e_hartree=E_H,
        e_hf_exchange=E_K_final,
        e_nuclear_repulsion=E_nn,
        e_total=e_total,
        grid=grid,
        e_xc=float(e_xc_final) + e_xc_aug,
        functional=str(functional),
        e_dft_plus_u=e_dft_plus_u_final,
    )

    # Diagnostic: Hartree augmentation from J_aug - J_gpw.
    gpw_J = gapw_builder._gpw.build_J(D_total)
    gapw_J_diag = gapw_builder.build_J(D_total)
    gapw_correction = 0.5 * float(np.einsum("ij,ij->", D_total, gapw_J_diag - gpw_J))

    return GapwUksScfResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=breakdown.e_total,
        breakdown=breakdown,
        density_alpha=D_alpha,
        density_beta=D_beta,
        mo_coeffs_alpha=C_alpha,
        mo_coeffs_beta=C_beta,
        mo_energies_alpha=e_a,
        mo_energies_beta=e_b,
        n_alpha=n_alpha_resolved,
        n_beta=n_beta_resolved,
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        gapw_correction=gapw_correction,
        one_centre=one_centre,
        e_dft_plus_u=e_dft_plus_u_final,
        dft_plus_u_sites=_dftu_sites,
        scf_trace=tuple(scf_trace),
        fock_alpha=F_alpha,
        fock_beta=F_beta,
        overlap=S,
        molecular_limit_declared=(one_centre == "analytic"),
    )


def _gapw_multik_occupations(
    eps_per_k,
    weights,
    n_electrons_per_cell: float,
    n_occ_each: int,
    smearing_T: float,
    smearing_opts,
):
    """Occupations for the multi-k GAPW SCF: ``(occ_per_k, mu, entropy)``.

    At ``T = 0`` this fills the lowest states across the COMPLETE k-mesh
    under one global Fermi level. The alternative -- filling the lowest
    ``n_occ_each`` bands at *each* k independently -- is not the Aufbau
    ground state whenever bands overlap between k-points, and it is what
    made multi-k GAPW fail to converge on expanded cells
    (MULTIK-GAPW-BLOCH-FAILS-ON-EXPANDED-CELLS). On LiH rocksalt/STO-3G/
    LDA/(2,2,2) the fingerprint was a NEGATIVE gap at 1.60x the
    equilibrium lattice constant -- ``max_k eps_2 > min_k eps_3`` -- so
    two bands per k occupied states that were not the globally lowest and
    the SCF chased an occupation pattern that changed every iteration.

    This mirrors what the multi-k GDF route already does; the same defect
    was fixed there on 2026-08-01 (PERIODIC-K6-NONCONV), whose in-source
    note records that per-k filling put LiH FCC (2,2,2) at -2.96 Ha
    against a documented -7.92 Ha. `_global_aufbau_with_mu` is that fix's
    helper, reused here rather than reimplemented.

    Gapped systems are unaffected: when no bands overlap, the global fill
    reproduces the per-k pattern exactly.
    """
    if float(smearing_T) > 0.0:
        sm = _apply_smearing(
            eps_per_k,
            weights=list(weights),
            n_electrons_per_cell=float(n_electrons_per_cell),
            n_occ_each=int(n_occ_each),
            smearing=smearing_opts,
        )
        return (
            [np.asarray(o, dtype=float) for o in sm.occupations_per_k],
            float(sm.mu),
            float(sm.entropy),
        )
    occ, mu = _global_aufbau_with_mu(
        eps_per_k, list(weights), float(n_electrons_per_cell)
    )
    return [np.asarray(o, dtype=float) for o in occ], float(mu), 0.0


def _occupations_are_per_k_integer(occ_per_k, n_occ_each: int) -> bool:
    """Whether every k carries the plain ``2[:n_occ], 0`` integer pattern.

    False means the global fill found band overlap (or a degenerate Fermi
    boundary) and produced occupations that a fixed-occupancy consumer --
    an analytic gradient, say -- may not assume.
    """
    n_occ = int(n_occ_each)
    for occ in occ_per_k:
        arr = np.asarray(occ, dtype=float)
        expected = np.zeros_like(arr)
        if n_occ > 0:
            expected[:n_occ] = 2.0
        if not np.allclose(arr, expected, rtol=0.0, atol=1e-12):
            return False
    return True


def run_periodic_rks_gapw_multi_k(
    system,
    basis,
    kmesh,
    *,
    functional: str,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 80,
    conv_tol_energy: float = 1e-8,
    conv_tol_density: float = 1e-6,
    damping: float = 0.0,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    dispersion: Optional[str] = None,
    dispersion_functional: Optional[str] = None,
    smearing_temperature: float = 0.0,
    smearing_method: str = "fermi-dirac",
    dft_plus_u_sites=None,
    initial_density_k: Optional[Sequence[np.ndarray]] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    quiet: bool = False,
    lmax: int = _DEFAULT_LMAX,
    soft_cutoff: float = _DEFAULT_SOFT_CUTOFF,
    n_radial: int = _DEFAULT_N_RADIAL,
    lebedev_order: int = _DEFAULT_LEBEDEV_ORDER,
) -> GpwMultiKScfResult:
    """Multi-k GAPW periodic RKS SCF (closed-shell, pure DFT only).

    All-electron GAPW version of :func:`run_periodic_rks_gpw_multi_k`.
    Uses the :class:`GapwJBuilder` for the GAPW-augmented Hartree J
    instead of the bare FFT-Poisson solve.

    Same restrictions as the GPW multi-k path:

    * **Closed-shell RKS only** -- no HF / UHF / UKS.
    * **Pure DFT functionals only** -- ``hf_exchange_fraction`` must be 0.
    * **S_k weighted occupation** -- every k-point gets the same
      number of doubly-occupied orbitals (no smearing).

    The SCF loop:
      1. Build T_lat, S_lat, V_ne_lat once.
      2. Build the GAPW J builder (per-atom augmentation grids).
      3. For each iter:
         a. Bloch-sum T, S, V_ne, F per k.
         b. Eigen-solve F(k) C(k) = e S(k) C(k) per k.
         c. Build r(r) on the FFT grid from C(k), w_k.
         d. GAPW J (with augmentation) and libxc V_xc on r.
         e. Update F(k) = Hcore(k) + J_GAPW + V_xc.
      4. Convergence: ΔE < tol AND ||r_new - r_old||_F < tol.

    Parameters
    ----------
    system, basis
        Standard periodic system + basis.
    kmesh
        :class:`BlochKMesh` with k-points and weights.
    functional
        XC functional name (required).
    grid, cutoff_ha
        FFT grid or cutoff.
    max_iter, conv_tol_energy, conv_tol_density
        SCF convergence parameters.
    v_ne_convention
        V_ne convention: "ewald" or "smeared_erfc".
    smearing_alpha
        Smearing exponent for the smeared V_ne (bohr\u207b\u00b9).
    dispersion, dispersion_functional
        D3-BJ dispersion correction (requires the ``dftd3`` python
        bindings; raises ``NotImplementedError`` when missing).
    smearing_temperature, smearing_method
        Fermi-Dirac smearing parameters.
    dft_plus_u_sites
        DFT+U Hubbard sites.
    quiet
        Suppress experimental warnings.
    lmax
        Multipole order for the r\u2080 compensator.
    soft_cutoff
        Exponent threshold for the softened basis (bohr\u207b\u00b2).
    n_radial
        Number of radial points per atomic grid.
    lebedev_order
        Lebedev angular grid order.

    Returns
    -------
    result
        :class:`GpwMultiKScfResult` with the converged GAPW-corrected
        all-electron energy.
    """
    input_restart_supplied = initial_density_k is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver="run_periodic_rks_gapw_multi_k",
        supported=periodic_guess_capabilities('gapw', 'RKS', dim=getattr(system, "dim", 3), multi_k=True, transport='k'),
        restart_supplied=initial_density_k is not None,
    )

    if not quiet:
        warnings.warn(
            "run_periodic_rks_gapw_multi_k: M3e experimental -- "
            "GAPW all-electron multi-k RKS. The augmentation correction "
            "is active; results are preliminary.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    if system.dim != 3:
        raise ValueError(
            f"run_periodic_rks_gapw_multi_k: dim == 3 only; got dim={system.dim}"
        )

    func = _core.Functional(functional, 1)
    if bool(getattr(func, "is_external", False)):
        raise NotImplementedError(
            "run_periodic_rks_gapw_multi_k: full-grid external XC is "
            "currently supported by GAPW at Gamma only. Use "
            "run_periodic_rks_gapw with a Gamma-point calculation; "
            "multi-k GAPW external XC remains gated."
        )
    if float(func.hf_exchange_fraction) != 0.0:
        raise NotImplementedError(
            f"run_periodic_rks_gapw_multi_k: hybrids "
            f"(hf_exchange_fraction = "
            f"{func.hf_exchange_fraction}) are not supported "
            f"on the multi-k GAPW path -- per-k K builders are not "
            f"wired. Use a pure functional like 'lda' or 'pbe'."
        )

    # k-mesh. Compact cells need Bloch real-space XC density/tau and per-k
    # smooth-potential projection; molecular-limit cells keep the legacy
    # Gamma-folded density convention.
    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = kpoints.shape[0]
    use_bloch_compact = n_k > 1 and not _multik_gpw_is_molecular_limit(system)

    # Set up grid.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_periodic_rks_gapw_multi_k: cell has {n_elec} "
            f"electrons (odd); RKS needs even count."
        )
    n_occ = n_elec // 2
    n_basis = basis.nbasis

    # Smearing.
    smearing_T = float(smearing_temperature)
    if smearing_T < 0.0:
        raise ValueError(
            "run_periodic_rks_gapw_multi_k: smearing_temperature must be "
            f">= 0; got {smearing_T}"
        )
    if smearing_T > 0.0:
        smearing_opts = _SmearingOptions(
            temperature=smearing_T,
            flavor=str(smearing_method),
        )
    else:
        smearing_opts = None

    # One-electron lattice integrals (built once).
    lat_opts = _core.LatticeSumOptions()
    T_lat = _core.compute_kinetic_lattice(basis, system, lat_opts)
    S_lat = _core.compute_overlap_lattice(basis, system, lat_opts)
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    lat_opts_v = _core.LatticeSumOptions()
    lat_opts_v.coulomb_method = _core.CoulombMethod.EWALD_3D
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts_v)

    # Nuclear repulsion.
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    # DFT+U setup (same multi-k pattern as run_periodic_rks_gpw_multi_k).
    _dftu_sites = list(dft_plus_u_sites or ())
    _dftu_sites_cxx: list = []
    _dftu_ao_groups_list: list = []
    if _dftu_sites:
        from ._vibeqc_core import _HubbardSiteCxx as _HSC
        from .dft_plus_u import ao_group_indices as _dftu_ao_group_indices

        _ao_map = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (_site.atom_index, _site.l)
            if _key not in _ao_map:
                raise ValueError(
                    f"run_periodic_rks_gapw_multi_k: HubbardSite{_key} "
                    f"has no AOs in the basis. Available channels: "
                    f"{sorted(_ao_map.keys())}"
                )
            _dftu_sites_cxx.append(
                _HSC(
                    _site.atom_index,
                    _site.l,
                    _site.U_eff_hartree,
                )
            )
            _dftu_ao_groups_list.append(_ao_map[_key])
    e_dft_plus_u = 0.0

    # GAPW J builder (k-independent, per-atom augmentation grids).
    gapw_builder = GapwJBuilder(
        basis,
        system,
        grid,
        lmax=lmax,
        soft_cutoff=soft_cutoff,
        n_radial=n_radial,
        lebedev_order=lebedev_order,
        quiet=quiet,
    )
    collocation_cache = gapw_builder._collocation_cache

    # Pre-bloch-sum to T(k), S(k), V_ne(k) at every k.
    T_k: list = []
    S_k: list = []
    V_ne_k: list = []
    X_k: list = []  # canonical orthogonalizer per k
    if use_bloch_compact:
        from .periodic_rhf_multi_k_ewald import (
            _canonical_orthogonalizer_complex,
        )
    n_lindep_dropped = 0
    for ik in range(n_k):
        k = kpoints[ik]
        Tk = np.asarray(_core.bloch_sum(T_lat, k))
        Sk = np.asarray(_core.bloch_sum(S_lat, k))
        Vk = np.asarray(_core.bloch_sum(V_lat, k))
        # Hermitise the bloch sums against tiny noise.
        Sk = 0.5 * (Sk + Sk.conj().T)
        Tk = 0.5 * (Tk + Tk.conj().T)
        Vk = 0.5 * (Vk + Vk.conj().T)
        if use_bloch_compact:
            X, n_kept = _canonical_orthogonalizer_complex(
                Sk, threshold=_COMPACT_GPW_LINDEP_THRESHOLD
            )
            n_lindep_dropped += Sk.shape[0] - n_kept
        else:
            # Canonical orthogonalizer X = S^{-1/2}.
            s_eigs, U = np.linalg.eigh(Sk)
            X = U @ np.diag(1.0 / np.sqrt(np.maximum(s_eigs, 1e-12))) @ U.conj().T
        T_k.append(Tk)
        S_k.append(Sk)
        V_ne_k.append(Vk)
        X_k.append(X)
    if use_bloch_compact and n_lindep_dropped and not quiet:
        warnings.warn(
            f"run_periodic_rks_gapw_multi_k: dropped {n_lindep_dropped} "
            f"linearly-dependent Bloch-AO direction(s) across the k-mesh "
            f"(S(k) eigenvalue < {_COMPACT_GPW_LINDEP_THRESHOLD:.0e}); "
            "compact crystal + molecular basis over-completeness.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    # Bloch-AO image reach. This is NOT a free parameter: the Bloch tables
    # satisfy
    #
    #     int_cell chi_mk chi_nk^*  =  S(k)
    #
    # EXACTLY when the sum over lattice translations T is complete, which is
    # what makes the collocated smooth density carry the true soft charge
    # sum_k w_k tr(D_s(k) S_s(k)). A hardcoded 25.0 bohr held here
    # regardless of cell size or basis, and on expanded cells it is not
    # enough: measured on LiH rocksalt/STO-3G/(2,2,2) at a FIXED density,
    # the collocated charge misses the exact value by -1.536e-01 e at
    # lattice scale 2.20, while a basis-sized reach converges it to
    # machine precision (-2.4e-14 at scale 1.00, -1.3e-15 at 1.60).
    #
    # That error is density-dependent, so inside the SCF it moves every
    # iteration -- the smooth charge swung 0.27 e between iterations on the
    # scale-1.60 cell while the exact charge held constant to 3 decimals --
    # and a fluctuating cell charge moves the G=0 gauge, which is why the
    # expanded cells never converged (MULTIK-GAPW-BLOCH-FAILS-ON-EXPANDED-CELLS).
    # Same defect family as GPW-MULTIK-OVERLAP-NOT-PSD: two independent
    # hardcoded reaches where one identity governs both.
    #
    # `bloch_overlap_cutoff_bohr` floors at 25.0, so no previously-adequate
    # cell shrinks its reach; it only grows where the basis demands it.
    #
    # The tolerance is deliberately looser than that helper's 1e-12 default,
    # which is sized to guarantee S(k) POSITIVE DEFINITENESS -- a stricter
    # requirement than converging a collocated charge. At 1e-12 the
    # equilibrium LiH cell would carry 1865 image cells against the previous
    # 555, a 3.4x cost for no change in the energy (25 bohr already gives
    # 1.5e-07 e there). At 1e-8 the reaches are 29.7 / 33.7 / 37.7 bohr over
    # the scale-1.00 / 1.60 / 2.20 scan, which lands the charge error at
    # 1e-9 or below everywhere while costing 1.7x at equilibrium.
    _BLOCH_AO_COLLOCATION_TOL = 1e-8
    from .lattice_screening import bloch_overlap_cutoff_bohr as _bloch_rcut

    _reach = _bloch_rcut(basis, system, tol=_BLOCH_AO_COLLOCATION_TOL)
    if gapw_builder._aug._augmentation_active:
        _reach = max(_reach, _bloch_rcut(
            gapw_builder._aug._soft_basis, system,
            tol=_BLOCH_AO_COLLOCATION_TOL))
    direct_cells = _core.direct_lattice_cells(system, float(_reach))
    lattice_translations = np.array([c.r_cart for c in direct_cells], dtype=float)
    grid_pts = grid.cartesian_coords().reshape(-1, 3)
    chi_k_list: list = []
    chi_gen_k_list: list = []
    if use_bloch_compact:
        for ik in range(n_k):
            chi_k_list.append(
                bloch_ao_on_grid(
                    basis, grid_pts, kpoints[ik], lattice_translations
                )
            )
        if gapw_builder._aug._augmentation_active:
            gen_basis = gapw_builder._aug._soft_basis
            for ik in range(n_k):
                chi_gen_k_list.append(
                    bloch_ao_on_grid(
                        gen_basis, grid_pts, kpoints[ik], lattice_translations
                    )
                )
        else:
            chi_gen_k_list = chi_k_list

    def _build_D_k(C_list, occ_list):
        return [
            (C_list[ik] * occ_list[ik][None, :]) @ C_list[ik].conj().T
            for ik in range(n_k)
        ]

    def _smooth_generation_D_k(D_k_list):
        if not gapw_builder._aug._augmentation_active:
            return D_k_list
        idx = gapw_builder._aug._soft_indices
        return [Dk[np.ix_(idx, idx)] for Dk in D_k_list]

    # Preserve the driver's global-Aufbau/smearing machinery for Fock-mode
    # guesses. Density-mode guesses enter through the shared periodic adapter
    # as a g=0 density block repeated over the Bloch mesh. Restart blocks win.
    guess_fock_k = None
    if initial_density_k is None and initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_full_hf_focks_k
        guess_fock_k, _ = patom_full_hf_focks_k(
            system, basis, kmesh, [t + v for t, v in zip(T_k, V_ne_k)],
            S_k, n_occ, n_occ, lat_opts)
    if initial_density_k is None and guess_fock_k is None:
        from .guess import initial_density_closed_shell, periodic_fock_guess_k

        guess_fock_k = periodic_fock_guess_k(
            system,
            basis,
            kpoints,
            initial_guess,
            lattice_opts=lat_opts,
            kinetic_lattice=T_lat,
            overlap_lattice=S_lat,
        )
        if guess_fock_k is None:
            density_g0 = initial_density_closed_shell(
                system.unit_cell_molecule(),
                basis,
                n_occ,
                initial_guess,
                is_periodic=True,
                periodic_system=system,
                lattice_opts=lat_opts,
                overlap=S_k, weights=weights,
            )
            if density_g0 is not None:
                initial_density_k = [
                    np.asarray(density_g0, dtype=complex).copy()
                    for _ in range(n_k)
                ]

    # Build the initial density from SAP or Hcore and accumulate rho. The
    # existing smearing/global-Aufbau code below owns the occupations.
    D_total = np.zeros((n_basis, n_basis), dtype=float)
    C_k: list = []
    e_k: list = []
    eps_per_k_init: list = []
    for ik in range(n_k):
        guess_fock = (
            guess_fock_k[ik]
            if guess_fock_k is not None
            else T_k[ik] + V_ne_k[ik]
        )
        eps, C_orth = np.linalg.eigh(
            X_k[ik].conj().T @ guess_fock @ X_k[ik]
        )
        C = X_k[ik] @ C_orth
        C_k.append(C)
        e_k.append(eps)
        eps_per_k_init.append(np.asarray(np.real(eps), dtype=float))
        # k-resolved 1-RDM contribution: D(k) = 2 \u00b7 C_occ C_occ\u2020 \u00b7 w_k
        Dk = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].conj().T)
        D_total += weights[ik] * np.real(Dk)
    D_total = 0.5 * (D_total + D_total.T)

    # Initial smearing pass.
    if smearing_T > 0.0:
        sm_init = _apply_smearing(
            eps_per_k_init,
            weights=list(weights),
            n_electrons_per_cell=float(n_elec),
            n_occ_each=n_occ,
            smearing=smearing_opts,
        )
        occ_per_k_curr = [np.asarray(o, dtype=float) for o in sm_init.occupations_per_k]
        D_total = np.zeros((n_basis, n_basis), dtype=float)
        for ik in range(n_k):
            Dk = (C_k[ik] * occ_per_k_curr[ik][None, :]) @ C_k[ik].conj().T
            D_total += weights[ik] * np.real(Dk)
        D_total = 0.5 * (D_total + D_total.T)
    else:
        # T = 0: fill the lowest states across the WHOLE k-mesh under one
        # global Fermi level, not the lowest n_occ at EACH k. The per-k
        # form silently mis-occupies any system whose bands overlap
        # between k-points -- see the module note on
        # `_gapw_multik_occupations` below.
        occ_per_k_curr, _mu_init = _global_aufbau_with_mu(
            eps_per_k_init, list(weights), float(n_elec)
        )
        occ_per_k_curr = [np.asarray(o, dtype=float) for o in occ_per_k_curr]
        D_total = np.zeros((n_basis, n_basis), dtype=float)
        for ik in range(n_k):
            Dk = (C_k[ik] * occ_per_k_curr[ik][None, :]) @ C_k[ik].conj().T
            D_total += weights[ik] * np.real(Dk)
        D_total = 0.5 * (D_total + D_total.T)
    D_k_curr = _build_D_k(C_k, occ_per_k_curr)
    if initial_density_k is not None:
        from .guess import normalize_density_k_guess
        D_k_curr = normalize_density_k_guess(initial_density_k, S_k, weights, n_elec)
        D_total = _weighted_real_density_from_k(D_k_curr, weights, n_basis)

    if n_k > 1 and not (
        use_bloch_compact and gapw_builder._aug._augmentation_active
    ):
        # WRONG-ANSWER GUARD (2026-08-01), now scoped to the paths the
        # Bloch port below does NOT cover: the compact + augmentation-active
        # case builds its smooth Hartree from the Bloch density and projects
        # it per k (``build_J_bloch``), so the fold is not used there.
        # Everything else still consumes the fold and must be checked. This driver routes only its XC
        # by regime; the Hartree term always goes through
        # ``gapw_builder.build_J(D_total)``, i.e. the Gamma-folded density
        # matrix D(g=0) collocated against Gamma AOs. The true periodic
        # density needs D(R'-R) per cell pair,
        #   rho(r) = sum_{munu,RR'} D_munu(R'-R) chi_mu(r-R) chi_nu(r-R'),
        # so reusing D(g=0) for EVERY pair is right only where the
        # cross-cell AO products vanish. Where they do not, the fold
        # double-counts inter-cell density.
        #
        # The guard measures the defect directly instead of trusting a
        # regime heuristic: collocate the fold and check it still carries
        # the cell's electron count. That distinction is not academic --
        # measured Gamma-fold charge error, STO-3G, Hcore-guess density:
        # He/8-bohr cube (1,1,2) 0.0 %, He/6-bohr (2,2,2) +0.5 %, LiH
        # rocksalt (2,2,2) **+145 %** (9.82 e for a 4-electron cell,
        # worth 340 mHa in E_H) -- while ``_multik_gpw_is_molecular_limit``
        # calls ALL THREE compact. Gating on that flag would have refused
        # two working configurations.
        #
        # The GPW sibling (``run_periodic_rks_gpw_multi_k``) routes J
        # correctly via ``_collocate_multik`` + per-k projection; porting
        # that here (including the augmentation's soft and compensator
        # pieces, which inherit the same fold) is the fix. Filed in
        # HANDOVER_OPEN_BUGS_V015.md; until it lands, refuse rather than
        # return a wrong number (CLAUDE.md section 7).
        _q_fold = float(
            collocate_density_on_grid(
                basis, D_total, grid, cache=collocation_cache
            ).sum()
        ) * grid.voxel_volume_bohr3
        _rel = (_q_fold - n_elec) / max(float(n_elec), 1.0)
        if _rel > _MULTIK_FOLD_CHARGE_TOL:
            raise NotImplementedError(
                "run_periodic_rks_gapw_multi_k: the Gamma-folded density "
                f"this driver feeds its Hartree build carries {_q_fold:.4f} "
                f"electrons, but the cell has {n_elec} "
                f"({100.0 * _rel:+.1f} %). On a cell whose AO products "
                "reach neighbouring cells, folding D(g=0) into every cell "
                "pair double-counts inter-cell density, so the Hartree "
                "term (and the augmentation built on it) would be wrong "
                "-- on LiH rocksalt (2,2,2) by 340 mHa. Multi-k GAPW "
                "supports only cells where the fold is valid (measured "
                f"tolerance {100.0 * _MULTIK_FOLD_CHARGE_TOL:.0f} %); "
                "vacuum-padded / molecular-limit cells qualify. For "
                "compact crystals use jk_method='gdf' (all-electron, "
                "PySCF-validated) or jk_method='gpw' (which routes its "
                "density by regime)."
            )

    scf_trace: list = []
    E_prev = 0.0
    converged = False
    n_iter = 0
    fermi_level = 0.0
    entropy = 0.0
    # Previous iteration's occupations, for the stability half of the
    # convergence test (see the check at the bottom of the loop).
    _occ_prev = None
    # Pulay DIIS over the per-k Fock list. Reuses the same adapter the
    # multi-k GDF route uses rather than re-deriving the k-weighted inner
    # product here. `use_diis=False` recovers the bare fixed-point
    # iteration this driver used to be.
    if use_diis and int(diis_subspace_size) >= 2:
        from .periodic_scf_accelerators import _MultiKPulayDIIS

        _diis = _MultiKPulayDIIS(int(diis_subspace_size))
    else:
        _diis = None

    for it in range(1, max_iter + 1):
        # GAPW Hartree J. Two regimes, mirroring the XC block below:
        #   * compact Bloch -- the smooth density must come from the Bloch
        #     sum (folding D(g=0) into every cell pair double-counts
        #     inter-cell density; see the fold-charge guard above), and the
        #     smooth potential is a LOCAL operator projected per k. The
        #     one-centre augmentation blocks stay k-independent: their
        #     atomic regions are non-overlapping by construction, so only
        #     same-cell image pairs reach inside them.
        #   * molecular limit -- the validated Gamma path, unchanged.
        J_smooth_per_k: list = [None] * n_k
        if use_bloch_compact and gapw_builder._aug._augmentation_active:
            rho_tilde_bloch = collocate_bloch_density_on_grid(
                chi_gen_k_list,
                _smooth_generation_D_k(D_k_curr),
                weights,
                grid,
            )
            V_smooth_bloch, J_ao, e_hartree = gapw_builder.build_J_bloch(
                D_total, rho_tilde_bloch
            )
            _sidx = gapw_builder._aug._soft_indices
            for ik in range(n_k):
                blk = project_potential_to_bloch_ao(
                    chi_gen_k_list[ik], V_smooth_bloch, grid
                )
                full = np.zeros((n_basis, n_basis), dtype=complex)
                full[np.ix_(_sidx, _sidx)] = blk
                J_smooth_per_k[ik] = full
        else:
            J_ao = gapw_builder.build_J(D_total)
            e_hartree = 0.5 * float(np.einsum("ij,ij->", D_total, J_ao))

        # GAPW smooth XC is generated from the soft density (when active), then
        # projected onto the full AO basis. Compact cells use Bloch density/tau
        # and per-k projection; molecular-limit cells keep the Gamma path.
        V_xc_per_k: list = [None] * n_k
        V_xc = None
        if use_bloch_compact:
            D_gen_k_curr = _smooth_generation_D_k(D_k_curr)
            rho_grid = collocate_bloch_density_on_grid(
                chi_gen_k_list, D_gen_k_curr, weights, grid
            )
            tau_bloch = None
            if func.kind == _core.XCKind.MGGA:
                tau_bloch = compute_bloch_kinetic_energy_density(
                    chi_gen_k_list,
                    D_gen_k_curr,
                    weights,
                    kpoints,
                    grid_pts,
                    grid,
                    recip=collocation_cache.recip,
                )
            e_xc_iter, v_eff_xc, v_tau_grid = _xc_effective_potential_grid(
                rho_grid,
                grid,
                func,
                tau=tau_bloch,
                cache=collocation_cache,
            )
            for ik in range(n_k):
                V_xc_per_k[ik] = project_potential_to_bloch_ao(
                    chi_k_list[ik], v_eff_xc, grid
                )
                if v_tau_grid is not None:
                    V_xc_per_k[ik] = V_xc_per_k[ik] + project_vtau_to_bloch_ao(
                        chi_k_list[ik],
                        kpoints[ik],
                        v_tau_grid,
                        grid_pts,
                        grid,
                        recip=collocation_cache.recip,
                    )
        else:
            _gb, _Dg, _gcc = gapw_builder._smooth_xc_generation(D_total)
            rho_grid = collocate_density_on_grid(_gb, _Dg, grid, cache=_gcc)
            (
                e_xc_iter,
                v_xc_grid,
                v_sigma_grid,
                grad_rho,
                v_tau_grid,
            ) = _evaluate_xc_on_grid(
                rho_grid,
                grid,
                func,
                basis=_gb,
                density_matrix=_Dg,
                cache=_gcc,
            )
            V_xc = _project_vxc_to_ao(
                basis,
                v_xc_grid,
                grid,
                v_sigma_grid=v_sigma_grid,
                grad_rho=grad_rho,
                cache=collocation_cache,
            )
            if v_tau_grid is not None:
                if _gb is basis:
                    V_tau = _project_vtau_to_ao(
                        basis, v_tau_grid, grid, cache=collocation_cache
                    )
                else:
                    V_tau = _project_vtau_to_ao(basis, v_tau_grid, grid)
                V_xc = V_xc + V_tau

        V_xc_aug = np.zeros((n_basis, n_basis), dtype=float)
        xc_corrections, e_xc_aug_iter = (
            gapw_builder._aug.compute_xc_correction(
                D_total,
                func,
                return_energy=True,
            )
        )
        for delta_Vxc in xc_corrections.values():
            V_xc_aug = V_xc_aug + delta_Vxc
        e_xc_total_iter = float(e_xc_iter) + float(e_xc_aug_iter)
        if not use_bloch_compact:
            V_xc = V_xc + V_xc_aug

        # DFT+U per-spin per-k Fock contribution.
        e_dft_plus_u = 0.0
        V_U_per_k = [None] * n_k
        if _dftu_sites_cxx:
            from ._vibeqc_core import (
                _compute_dft_plus_u_multi_k_per_spin_cxx,
            )

            P_sigma_k: list = []
            for ik in range(n_k):
                Pk = (C_k[ik] * occ_per_k_curr[ik][None, :]) @ C_k[ik].conj().T
                P_sigma = 0.5 * Pk
                P_sigma = 0.5 * (P_sigma + P_sigma.conj().T)
                P_sigma_k.append(np.asarray(P_sigma, dtype=complex))
            S_k_cplx = [np.asarray(s, dtype=complex) for s in S_k]
            E_sigma, V_AO = _compute_dft_plus_u_multi_k_per_spin_cxx(
                _dftu_sites_cxx,
                _dftu_ao_groups_list,
                S_k_cplx,
                P_sigma_k,
                list(weights),
            )
            e_dft_plus_u = 2.0 * float(E_sigma)
            V_AO_c = np.asarray(V_AO, dtype=complex)
            for ik in range(n_k):
                V_U_per_k[ik] = S_k[ik] @ V_AO_c @ S_k[ik]

        # Per-k Fock, then (optionally) DIIS, then diagonalisation.
        F_k_list: list = []
        for ik in range(n_k):
            Hcore_k = T_k[ik] + V_ne_k[ik]
            if use_bloch_compact:
                Fk = Hcore_k + J_ao + V_xc_per_k[ik] + V_xc_aug
                if J_smooth_per_k[ik] is not None:
                    # Bloch regime: J_ao holds only the k-independent
                    # one-centre augmentation blocks; the smooth Hartree
                    # term is the per-k projection of V_smooth.
                    Fk = Fk + J_smooth_per_k[ik]
            else:
                Fk = Hcore_k + J_ao + V_xc
            if V_U_per_k[ik] is not None:
                Fk = Fk + V_U_per_k[ik]
            F_k_list.append(0.5 * (Fk + Fk.conj().T))

        # Pulay DIIS on the per-k Fock list. Error vector per k is the
        # commutator e(k) = F(k) D(k) S(k) - S(k) D(k) F(k), which
        # vanishes exactly at self-consistency; the adapter maps the
        # per-k complex list onto sqrt(w_k)-scaled real blocks so the
        # canonical C++ kernel's inner product IS the k-weighted one.
        # Same object multi-k GDF uses -- this driver previously had no
        # Fock extrapolation of any kind.
        if _diis is not None and it >= diis_start_iter:
            err_k_list = []
            for ik in range(n_k):
                Fk = F_k_list[ik]
                Dk = D_k_curr[ik]
                Sk = S_k[ik]
                e_commutator = Fk @ Dk @ Sk - Sk @ Dk @ Fk
                # Transform to the orthonormal basis so the residual norm
                # is metric-independent (X^H e X), matching the Gamma
                # driver's S^-1/2 e S^-1/2 convention.
                err_k_list.append(
                    X_k[ik].conj().T @ e_commutator @ X_k[ik]
                )
            F_k_list = _diis.extrapolate(
                F_k_list, err_k_list, list(weights)
            )

        new_C_k = []
        new_e_k = []
        eps_per_k_new: list = []
        for ik in range(n_k):
            Fk = F_k_list[ik]
            eps, C_orth = np.linalg.eigh(X_k[ik].conj().T @ Fk @ X_k[ik])
            C = X_k[ik] @ C_orth
            new_C_k.append(C)
            new_e_k.append(eps)
            eps_per_k_new.append(np.asarray(np.real(eps), dtype=float))

        # Occupations. At T = 0 this is a GLOBAL Brillouin-zone Aufbau
        # fill (one Fermi level over the whole mesh); at finite T the
        # smearing package owns it. See `_gapw_multik_occupations`.
        occ_per_k_new, fermi_level, entropy = _gapw_multik_occupations(
            eps_per_k_new,
            list(weights),
            float(n_elec),
            int(n_occ),
            float(smearing_T),
            smearing_opts,
        )

        # Build D from fractional occupations.
        D_k_new = _build_D_k(new_C_k, occ_per_k_new)
        D_new = np.zeros_like(D_total)
        for ik in range(n_k):
            D_new += weights[ik] * np.real(D_k_new[ik])
        D_new = 0.5 * (D_new + D_new.T)

        # Linear density damping, same convention as the Gamma driver
        # (`D <- (1-a) D_new + a D_old`). The multi-k GAPW SCF was a bare
        # Roothaan fixed-point iteration with NO mixing of any kind --
        # unlike multi-k GDF (DIIS + dynamic damping + accelerators) and
        # unlike the Gamma GAPW driver (DIIS). An undamped fixed point
        # converges only while the iteration is contractive, which is why
        # this route converged on the gapped equilibrium cell and not on
        # the expanded ones. Default 0.0 keeps every existing run
        # bit-identical.
        if damping > 0.0:
            a = float(damping)
            D_new = (1.0 - a) * D_new + a * D_total
            D_k_new = [
                (1.0 - a) * dn + a * dc
                for dn, dc in zip(D_k_new, D_k_curr)
            ]

        # Total energy.
        e_hcore = 0.0
        for ik in range(n_k):
            Hcore_k = T_k[ik] + V_ne_k[ik]
            e_hcore += weights[ik] * float(
                np.real(np.einsum("ij,ji->", D_k_curr[ik], Hcore_k))
            )
        E = e_hcore + e_hartree + e_xc_total_iter + E_nn + e_dft_plus_u
        occ_per_k_curr = occ_per_k_new

        dE = E - E_prev
        dD = float(np.linalg.norm(D_new - D_total))
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_total_iter),
            }
        )
        # Occupations must ALSO have settled. With one global Fermi level
        # the occupied set can move between k-points while the density
        # norm barely changes, so an energy+density test alone can call
        # convergence on an iterate whose filling is still shifting.
        # `_occ_prev` is None on the first pass, which `it > 1` already
        # excludes.
        occ_stable = _occ_prev is not None and all(
            np.allclose(a, b, rtol=0.0, atol=1e-10)
            for a, b in zip(occ_per_k_new, _occ_prev)
        )
        _occ_prev = [np.array(o, copy=True) for o in occ_per_k_new]
        if (
            abs(dE) < conv_tol_energy
            and dD < conv_tol_density
            and occ_stable
            and it > 1
        ):
            converged = True
            D_total = D_new
            D_k_curr = D_k_new
            C_k = new_C_k
            e_k = new_e_k
            break
        D_total = D_new
        D_k_curr = D_k_new
        C_k = new_C_k
        e_k = new_e_k
        E_prev = E

    # Build a breakdown at the converged density.
    e_kinetic_mk = 0.0
    e_ne_mk = 0.0
    for ik in range(n_k):
        Dk = np.asarray(D_k_curr[ik], dtype=complex)
        e_kinetic_mk += weights[ik] * float(
            np.real(np.einsum("ij,ji->", Dk, T_k[ik]))
        )
        e_ne_mk += weights[ik] * float(
            np.real(np.einsum("ij,ji->", Dk, V_ne_k[ik]))
        )
    if use_bloch_compact and gapw_builder._aug._augmentation_active:
        # Same regime split as the SCF loop: the smooth half of the GAPW
        # Hartree energy must come from the Bloch density, not the fold.
        _, _, e_hartree_final = gapw_builder.build_J_bloch(
            D_total,
            collocate_bloch_density_on_grid(
                chi_gen_k_list,
                _smooth_generation_D_k(D_k_curr),
                weights,
                grid,
            ),
        )
    else:
        e_hartree_final = gapw_builder.gapw_hartree_energy(D_total)
    if use_bloch_compact:
        D_gen_k_final = _smooth_generation_D_k(D_k_curr)
        rho_grid_final = collocate_bloch_density_on_grid(
            chi_gen_k_list,
            D_gen_k_final,
            weights,
            grid,
        )
        tau_bloch_final = None
        if func.kind == _core.XCKind.MGGA:
            tau_bloch_final = compute_bloch_kinetic_energy_density(
                chi_gen_k_list,
                D_gen_k_final,
                weights,
                kpoints,
                grid_pts,
                grid,
                recip=collocation_cache.recip,
            )
        e_xc_smooth_final, _, _ = _xc_effective_potential_grid(
            rho_grid_final,
            grid,
            func,
            tau=tau_bloch_final,
            cache=collocation_cache,
        )
    else:
        _gb, _Dg, _gcc = gapw_builder._smooth_xc_generation(D_total)
        rho_grid_final = collocate_density_on_grid(
            _gb,
            _Dg,
            grid,
            cache=_gcc,
        )
        e_xc_smooth_final, *_ = _evaluate_xc_on_grid(
            rho_grid_final,
            grid,
            func,
            basis=_gb,
            density_matrix=_Dg,
            cache=_gcc,
        )
    _e_h_aug, e_xc_aug = gapw_builder._aug.compute_augmentation_energy(
        D_total,
        functional=func,
    )
    e_xc_total = float(e_xc_smooth_final) + float(e_xc_aug)
    E = e_kinetic_mk + e_ne_mk + e_hartree_final + e_xc_total + E_nn
    breakdown = GpwEnergyBreakdown(
        e_kinetic=e_kinetic_mk,
        e_nuclear_attraction=e_ne_mk,
        e_hartree=e_hartree_final,
        e_hf_exchange=0.0,
        e_nuclear_repulsion=E_nn,
        e_total=E,
        grid=grid,
        e_xc=e_xc_total,
        functional=functional,
    )

    # D3-BJ periodic dispersion.
    e_disp = _compute_d3_correction(
        system,
        dispersion,
        dispersion_functional,
    )
    if e_disp != 0.0:
        from dataclasses import replace as _dc_replace

        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_disp,
            e_dispersion=e_disp,
        )
        E = E + e_disp

    # DFT+U bookkeeping at the accepted (possibly damped) per-k density.
    e_dft_plus_u_final = 0.0
    if _dftu_sites_cxx:
        from dataclasses import replace as _dc_replace
        from ._vibeqc_core import (
            _compute_dft_plus_u_multi_k_per_spin_cxx,
        )

        E_sigma_final, _ = _compute_dft_plus_u_multi_k_per_spin_cxx(
            _dftu_sites_cxx,
            _dftu_ao_groups_list,
            [np.asarray(s, dtype=complex) for s in S_k],
            [0.5 * np.asarray(Dk, dtype=complex) for Dk in D_k_curr],
            list(weights),
        )
        e_dft_plus_u_final = 2.0 * float(E_sigma_final)

        E = E + e_dft_plus_u_final
        breakdown = _dc_replace(
            breakdown,
            e_total=breakdown.e_total + e_dft_plus_u_final,
            e_dft_plus_u=e_dft_plus_u_final,
        )

    return GpwMultiKScfResult(
        restart_kpoints=np.asarray(kmesh.kpoints).copy(),
        restart_weights=np.asarray(kmesh.weights).copy(),
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        guess_selection=periodic_result_selection(
            system, requested_initial_guess, restarted=input_restart_supplied),
        energy=E,
        breakdown=breakdown,
        mo_coeffs_k=tuple(C_k),
        mo_energies_k=tuple(e_k),
        density=D_total,
        density_k=tuple(np.asarray(Dk, dtype=complex) for Dk in D_k_curr),
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        kmesh=kmesh,
        overlap_k=tuple(np.asarray(s, dtype=complex).copy() for s in S_k),
        scf_trace=tuple(scf_trace),
        e_dispersion=e_disp,
        e_dft_plus_u=e_dft_plus_u_final,
        smearing_temperature=smearing_T,
        smearing_entropy=entropy,
        fermi_level=fermi_level,
        occupations_k=tuple(tuple(o.tolist()) for o in occ_per_k_curr),
    )
